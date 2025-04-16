#coding=utf8
import math, dgl, torch
import pickle
import json
import numpy as np
import re
# from utils.constants import MAX_RELATIVE_DIST
from .graph_example import GraphExample
from . import graph_example


special_column_mapping_dict = {
    'question-*-generic': 'question-column-nomatch',
    '*-question-generic': 'column-question-nomatch',
    'question-question-dist-1': 'question-question-dist1',
    'table-*-generic': 'table-column-has',
    '*-table-generic': 'column-table-has',
    '*-column-generic': 'column-column-generic',
    'column-*-generic': 'column-column-generic',
    '*-*-identity': 'column-column-identity'
}

all_relations = ['table-table-generic','table-table-fkb','table-table-fk','table-table-fkr','table-table-identity','column-column-generic','column-column-sametable','column-column-identity',
                 'column-column-fk','column-column-fkr','column-table-has','table-column-has','column-table-pk', 'table-column-pk','question-question-identity', 'question-question-generic', 
                 'question-question-dist1','question-table-nomatch','table-question-nomatch','question-table-exactmatch','table-question-exactmatch','question-table-partialmatch','table-question-partialmatch',
                 'question-column-nomatch','column-question-nomatch','question-column-exactmatch','column-question-exactmatch','question-column-partialmatch','column-question-partialmatch',
                 'question-column-valuematch','column-question-valuematch','question-question-modifier','question-question-argument']

nonlocal_relations = [
    'question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic',
    'question-question-identity', 'table-table-identity',  'symbol', '0', '*-*-identity',
    'column-column-identity', 'question-table-nomatch', 'question-column-nomatch', 'column-question-nomatch', 'table-question-nomatch']

upos_to_id = {
    "ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5,
    "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11,
    "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16, "SPACE": 17
}

components_list = [
    'select', 'multi-select', 'from', 'join', 'multi-join', 'on', 'where', 'where-and', 'where-or', 
    'where-between', 'where-like', 'where-in', 'cast', 'case', 'union',  'group by', 'order by', 
    'having', 'distinct', 'desc', 'asc', 'limit', 'count', 'sum', 'avg', 'min', 'max'
]
print(len(all_relations))

def analyze_sql_structure(sql_query):
    sql_components = {
        'select': r'\bselect\b',
        'multi-select': r'\bselect\b.*?\bselect\b',
        'from': r'\bfrom\b',
        'join': r'\bjoin\b',
        'multi-join': r'\bjoin\b.*?\bjoin\b',
        'on': r'\bon\b',  
        'where': r'\bwhere\b',
        'where-and': r'\bwhere\b.*?\band\b',  
        'where-or': r'\bwhere\b.*?\bor\b',   
        'where-between': r'\bwhere\b.*?\bbetween\b', 
        'where-like': r'\bwhere\b.*?\blike\b',      
        'where-in': r'\bwhere\b.*?\bin\b',         
        'cast': r'\bcast\b',
        'case': r'\bcase\b',
        'union': r'\bunion\b',
        'group by': r'\bgroup\s+by\b',
        'order by': r'\border\s+by\b',
        'having': r'\bhaving\b',
        'distinct': r'\bdistinct\b',
        'desc': r'\bdesc\b',
        'asc': r'\basc\b',
        'limit': r'\blimit\b',
        'count': r'\bcount\s*\(',
        'sum': r'\bsum\s*\(',
        'avg': r'\bavg\s*\(',
        'min': r'\bmin\s*\(',
        'max': r'\bmax\s*\('
    }
    
    result = {key: False for key in sql_components}

    for component, pattern in sql_components.items():
        if re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL):
            result[component] = True

    if result['where-and']:
        where_clause = re.search(r'\bwhere\b(.*)', sql_query, re.IGNORECASE | re.DOTALL)
        if where_clause:
            where_content = where_clause.group(1)
            all_and_matches = re.findall(r'\band\b', where_content, re.IGNORECASE)
            between_matches = re.findall(r'\bbetween\b.*?\band\b.*?', where_content, re.IGNORECASE)
            if len(all_and_matches) == len(between_matches):
                result['where-and'] = False
    
    if any(result[key] for key in ['where-and', 'where-or', 'where-between', 'where-like', 'where-in']):
        result['where'] = False

    return result

def sql_to_vector(analysis_result, components_list):
    vector = np.zeros(len(components_list))  
    
    for i, component in enumerate(components_list):
        if analysis_result.get(component, False):
            vector[i] = 1  
    
    return vector

def sql_to_vectors_pipeline(sql_query, components_list):
    analyze_result = analyze_sql_structure(sql_query)
    vector = sql_to_vector(analyze_result, components_list)
    return vector

def edge_attr_to_one_hot(relation: str):
    num_relations = len(all_relations)
    one_hot = np.zeros(num_relations, dtype=int)
    idx = all_relations.index(relation) if relation in all_relations else -1
    if idx >= 0:
        one_hot[idx] = 1
    return one_hot

def check_node_to_one_hot(i, upos_tags, q_num):
    num_classes = len(upos_to_id)
    one_hot = np.zeros(num_classes, dtype=int)

    if i < q_num:  
        upos_tag = upos_tags[i]
        idx = upos_to_id.get(upos_tag, -1)
        if idx >= 0:
            one_hot[idx] = 1
    else:
        one_hot = np.zeros(num_classes, dtype=int)
        one_hot[11] = 1 
    return one_hot



class GraphProcessor():

    def process_rgatsql(self, ex: dict, db: dict, train_dataset:dict,relation: list ):
        
        graph = GraphExample()
        upos_tags = ex['upos_tags']
        num_nodes = int(math.sqrt(len(relation)))
        print(num_nodes)
        local_edges = [(idx // num_nodes, idx % num_nodes, (special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
            for idx, r in enumerate(relation) if r not in nonlocal_relations]
        nonlocal_edges = [(idx // num_nodes, idx % num_nodes, (special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
            for idx, r in enumerate(relation) if r in nonlocal_relations]
        # global_edges_raw = local_edges + nonlocal_edges
        # local_edges = [(idx // num_nodes, idx % num_nodes, edge_attr_to_one_hot(special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
        #     for idx, r in enumerate(relation) if r not in nonlocal_relations]
        # nonlocal_edges = [(idx // num_nodes, idx % num_nodes, edge_attr_to_one_hot(special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
        #     for idx, r in enumerate(relation) if r in nonlocal_relations]
        global_edges = local_edges + nonlocal_edges
        print(len(global_edges))
        src_ids, dst_ids = list(map(lambda r: r[0], global_edges)), list(map(lambda r: r[1], global_edges))
        graph.global_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.global_edges = global_edges
        src_ids, dst_ids = list(map(lambda r: r[0], local_edges)), list(map(lambda r: r[1], local_edges))
        graph.local_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.local_edges = local_edges
        # graph pruning for nodes
        q_num = len(ex['processed_question_toks'])
        s_num = num_nodes - q_num
        graph.question_mask = [1] * q_num + [0] * s_num
        graph.schema_mask = [0] * q_num + [1] * s_num
        graph.gp = dgl.heterograph({
            ('question', 'to', 'schema'): (list(range(q_num)) * s_num,
            [i for i in range(s_num) for _ in range(q_num)])
            }, num_nodes_dict={'question': q_num, 'schema': s_num}, idtype=torch.int32
        )
        graph.node_features = torch.tensor(list(map(lambda i: check_node_to_one_hot(i, upos_tags, q_num), range(num_nodes))), dtype=torch.int32)
        for question_dict in train_dataset:
            if question_dict['question_id'] == ex['question_id']:
                print(f"question_dict: {question_dict}")
                graph.target_label = sql_to_vectors_pipeline(question_dict['SQL'], components_list=components_list)
        ex['graph'] = graph
        return ex


    def process_graph_utils(self, ex: dict, db: dict, train_dataset:dict, method: str = 'rgatsql', ):
        """ Example should be preprocessed by self.pipeline
        """
        q = np.array(ex['relations'], dtype='<U100')
        s = np.array(db['relations'], dtype='<U100')
        q_s = np.array(ex['schema_linking'][0], dtype='<U100')
        s_q = np.array(ex['schema_linking'][1], dtype='<U100')
        relation = np.concatenate([
            np.concatenate([q, q_s], axis=1),
            np.concatenate([s_q, s], axis=1)
        ], axis=0)
        relation = relation.flatten().tolist()
        ex = self.process_rgatsql(ex, db, train_dataset=train_dataset,relation=relation)
        return ex


dev_data_path = r"Data\dev_databases\dev_data.json"
dev_schema_path = r"Data\dev_databases\dev_schema.json"
dev_path = r"Data\dev_databases\dev.json"
output_path = r"Data\dev_databases\dev_processed_final.bin"

with open(dev_data_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

with open(dev_schema_path, "r", encoding="utf-8") as f:
    dev_schema = json.load(f)

with open(dev_path, "r", encoding="utf-8") as f:
    dev = json.load(f)

processor = GraphProcessor()

print(f"dev_data:{dev_data[0]}")
print(f"dev_schema:{dev_schema[0]}")
print(f"dev:{dev[0]}")
ex_list=[]
for i in dev_data:
    for j in dev_schema:
        if i['db_id']==j['db_id']:
            ex=processor.process_graph_utils(i,j ,train_dataset=dev)
            ex_list.append(ex)
    print(f"process question:{i['question']} for db_id:{i['db_id']}")
    
if output_path is not None:
    pickle.dump(ex_list, open(output_path, "wb"))
    print(f"Processed examples saved to {output_path}")

