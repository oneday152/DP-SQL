import argparse

def init_args():
    parser = argparse.ArgumentParser(description="Training Configuration")

    # General settings
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_epoch', type=int, default=50, help='Maximum number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--testing', action='store_true', help='Flag to indicate testing mode')
    parser.add_argument('--model', default='rgatsql', help='which text2sql model to use')
    parser.add_argument('--output_path', default=r"Graphix/", help='file path to save the best model')
    parser.add_argument('--eval_after_epoch', type=int, default=10, help='eval after epoch')
    parser.add_argument('--relation_share_heads', type=bool, default=True,help='Whether to share heads across relations')

    # Model hyperparameters
    parser.add_argument('--input_dim', type=int, default=18, help='Input dimension size')
    parser.add_argument('--gnn_hidden_size', type=int, default=256, help='Hidden size for GNN layers')
    parser.add_argument('--feat_drop', type=float, default=0.2, help='Feature dropout rate')
    parser.add_argument('--output_size', type=int, default=27, help='Output dimension size')
    parser.add_argument('--gnn_num_layers', type=int, default=8, help='Number of GNN layers')
    parser.add_argument('--relation_share_layers', type=bool, default=True, help='Whether to share layers across relations')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in GNN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for model layers')

    # New parameter
    parser.add_argument('--local_and_nonlocal', type=str, default='local', help='Use local and global')

    args = parser.parse_args()
    return args