#coding=utf8
import sys, os, time, json, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from argparse import Namespace
from utils.args import init_args
from utils.example import Example
import utils.graph_example
from utils.graph_example import GraphFactory,GraphExample
from utils.set_logger import set_logger
from utils.batch import Batch
from model.model_utils import Registrable
from model.encoder.graph_encoder import *
import torch
import torch.optim as optim

args = init_args()
exp_path = r"Data\dev_databases"
logger = set_logger(exp_path, args.testing)
# set_random_seed(args.seed)
device = args.device
print()
start_time = time.time()
Example.configuration(method='rgatsql',choice='train')
train_dataset  = Example.load_dataset('train')
Example.configuration(method='rgatsql',choice='dev')
dev_dataset = Example.load_dataset('dev')
logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
args.relation_num = len(Example.relation_vocab)
print(args.relation_num)

# model init, set optimizer
model = Registrable.by_name('encoder_text2sql')(params, sql_trans).to(device)


def train_model_with_evaluation(model, train_dataset, dev_dataset, args, logger):
    if not args.testing:
        num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
        logger.info('Total training steps: %d' % num_training_steps)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        criterion = nn.BCEWithLogitsLoss()

        start_epoch, nsamples = 0, len(train_dataset)
        train_index = np.arange(nsamples)

        best_dev_acc = 0.0 
        best_epoch = -1    
        model_save_path = os.path.join(args.output_path, 'best_model.bin')  

        logger.info('Start training ......')
        for epoch in range(start_epoch, args.max_epoch):
            start_time = time.time()
            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0

            np.random.shuffle(train_index)
            model.train()
            for j in range(0, nsamples, args.batch_size):
                cur_dataset = [train_dataset[k] for k in train_index[j: j + args.batch_size]]
                current_batch = Batch.from_example_list(cur_dataset, device, train=True)
                
                outputs = model(current_batch)

                loss = criterion(outputs, current_batch.labels.float()) 
                epoch_loss += loss.item()

                predictions = torch.sigmoid(outputs) > 0.5  
                correct_predictions += (predictions == current_batch.graph.target_label).sum().item()
                total_predictions += current_batch.graph.target_label.numel()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            average_loss = epoch_loss / nsamples
            accuracy = correct_predictions / total_predictions

            logger.info('Epoch: %d\tTime: %.4f\tTraining Loss: %.4f\tAccuracy: %.4f' % 
                        (epoch, time.time() - start_time, average_loss, accuracy))
            torch.cuda.empty_cache()
            gc.collect()

            if epoch >= args.eval_after_epoch:
                logger.info("Evaluating on dev set at epoch %d ..." % epoch)
                dev_acc = evaluate_model(model, dev_dataset, args, logger)

                if dev_acc > best_dev_acc:  
                    best_dev_acc = dev_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_save_path)
                    logger.info("New best model saved at epoch %d with dev_acc: %.4f" % (epoch, dev_acc))

        logger.info("Training finished. Best model saved from epoch %d with dev_acc: %.4f" % (best_epoch, best_dev_acc))


def evaluate_model(model, dataset, args, logger):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
            outputs = model(current_batch)
            predictions = torch.sigmoid(outputs) > 0.5
            total_correct += (predictions == current_batch.graph.target_label).sum().item()
            total_samples += current_batch.graph.target_label.numel()

    dev_acc = total_correct / total_samples
    logger.info("Dev set accuracy: %.4f" % dev_acc)
    return dev_acc

