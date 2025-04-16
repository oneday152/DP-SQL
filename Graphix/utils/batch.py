#coding=utf8
import torch
import sys
import numpy as np
from Graphix.utils.example import Example
from Graphix.utils.graph_example import GraphFactory,GraphExample
from Graphix.utils.constants import PAD, UNK
import torch.nn.functional as F

def from_example_list_base(ex_list, device='gpu'):
    """
        question_lens: torch.long, bsize
        questions: torch.long, bsize x max_question_len, include [CLS] if add_cls
        table_lens: torch.long, bsize, number of tables for each example
        table_word_lens: torch.long, number of words for each table name
        tables: torch.long, sum_of_tables x max_table_word_len
        column_lens: torch.long, bsize, number of columns for each example
        column_word_lens: torch.long, number of words for each column name
        columns: torch.long, sum_of_columns x max_column_word_len
    """
    batch = Batch(ex_list, device)
    return batch

def from_example_list_text2sql(ex_list, device='cpu',is_tool=False, **kwargs):
    """ New fields: batch.lens, batch.max_len, batch.relations, batch.relations_mask
    """
    batch = from_example_list_base(ex_list, device)
    print(f"Example module ID from batch: {id(Example)}")
    print(f"Example module path: {sys.modules['Graphix.utils.example'].__file__}")
    print(f"Example.graph_factory: {Example.graph_factory}")
    batch.graph = Example.graph_factory.batch_graphs(ex_list, device,is_tool=is_tool, **kwargs)
    # if train:
    #     batch.max_action_num = max([len(ex.tgt_action) for ex in ex_list])
    return batch

class Batch():

    def __init__(self, examples, device='gpu'):
        super(Batch, self).__init__()
        self.examples = examples
        self.device = device

    @classmethod
    def from_example_list(cls, ex_list, device='gpu', train=True, method='text2sql', is_tool=False, **kwargs):
        method_dict = {
            "text2sql": from_example_list_text2sql,
        }
        return method_dict[method](ex_list, device, is_tool=is_tool, **kwargs)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

