#coding=utf8
import os, pickle, json
import sys
import torch, random
import numpy as np
from Graphix.utils import graph_example
from Graphix.utils.graph_example import GraphFactory,GraphExample
from Graphix.utils.constants import RELATIONS
from Graphix.utils.vocab import Vocab


class Example():

    @classmethod
    def configuration(cls,method='rgatsql',choice='train' ):
        assert choice in ['train','dev']
        if choice =='train':
            table_path=r"Data\dev_databases\dev_tables.json"
        else:
            table_path=r"Data\dev_databases\dev_tables.json"
        with open(table_path, "r") as f:
            cls.tables = json.load(f)
        cls.method = method
        cls.relation_vocab = Vocab(padding=False, unk=False, boundary=False, iterable=RELATIONS, default=None)
        cls.graph_factory = GraphFactory(cls.method, cls.relation_vocab)
        # print(f"graph_factory initialized: {cls.graph_factory}")

    @classmethod
    def load_dataset(cls, choice, debug=False):
        if choice not in ['train', 'dev']:
            raise ValueError("choice must be 'train' or 'dev'")

        if choice == 'train':
            fp = r"D:\dev_databases\dev_processed_final.bin"
            print("Loading training dataset...")
        else:
            fp = r"D:\dev_databases\dev_processed_final.bin"
            print("Loading development dataset...")

        with open(fp, 'rb') as file:
            print(sys.path)
            print(graph_example)
            datasets = pickle.load(file)

        examples, outliers = [], 0

        for ex in datasets:
            try:
                db = next((db for db in cls.tables if db['db_id'] == ex['db_id']))
            except StopIteration:
                continue

            if choice == 'train' and len(db['column_names']) > 200:
                outliers += 1
                continue

            examples.append(cls(ex, db, is_tool=False))

            if debug and len(examples) >= 100:
                return examples

        if choice == 'train':
            print("Skip %d extremely large samples in training dataset ..." % (outliers))

        return examples
    
    @classmethod
    def load_dataset_for_tool(cls, dataset,is_tool=True):
        example=[]
        for ex in dataset:
            try:
                db = next((db for db in cls.tables if db['db_id'] == ex['db_id']))
            except StopIteration:
                continue
            example.append(cls(ex, db, is_tool=True))
        return example

    def __init__(self, ex: dict, db: dict,method='rgatsql', choice='train', is_tool= False):
        super(Example, self).__init__()
        self.ex = ex
        self.db = db
        if  is_tool:
            self.graph = Example.graph_factory.graph_construction(ex, db, is_tool=True)
        else:
            self.graph = Example.graph_factory.graph_construction(ex, db, is_tool=False)