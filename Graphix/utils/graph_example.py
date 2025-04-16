#coding=utf8
import numpy as np
import dgl, torch, math

class GraphExample():

    pass

class BatchedGraph():

    pass

class GraphFactory():

    def __init__(self, method='rgatsql', relation_vocab=None):
        super(GraphFactory, self).__init__()
        self.method = eval('self.' + method)
        self.batch_method = eval('self.batch_' + method)
        self.relation_vocab = relation_vocab

    def graph_construction(self, ex: dict, db: dict,is_tool=False):
        if is_tool:
            return self.method(ex, db, is_tool=True)
        else:
            return self.method(ex, db, is_tool=False)

    def rgatsql(self, ex, db, is_tool=False):
        graph = GraphExample()
        local_edges = ex['graph'].local_edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], local_edges))
        graph.local_edges = torch.tensor(rel_ids, dtype=torch.long)
        global_edges = ex['graph'].global_edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], global_edges))
        graph.global_edges = torch.tensor(rel_ids, dtype=torch.long)
        graph.local_g, graph.global_g = ex['graph'].local_g, ex['graph'].global_g
        graph.gp = ex['graph'].gp
        graph.question_mask = torch.tensor(ex['graph'].question_mask, dtype=torch.bool)
        graph.schema_mask = torch.tensor(ex['graph'].schema_mask, dtype=torch.bool)
        graph.node_features = torch.tensor(ex['graph'].node_features, dtype=torch.float)
        if is_tool==False:
            graph.target_label = torch.tensor(ex['graph'].target_label, dtype=torch.float)
        # extract local relations (used in msde), global_edges = local_edges + nonlocal_edges
        local_enum, global_enum = graph.local_edges.size(0), graph.global_edges.size(0)
        graph.local_mask = torch.tensor([1] * local_enum + [0] * (global_enum - local_enum), dtype=torch.bool)
        return graph


    def batch_graphs(self, ex_list, device, train=True,is_tool=False, **kwargs):
        """ Batch graphs in example list """
        if is_tool:
            return self.batch_method(ex_list, device, train=train, is_tool=True, **kwargs)
        else:
            return self.batch_method(ex_list, device, train=train, is_tool=False, **kwargs)


    def batch_rgatsql(self, ex_list, device, train=True, is_tool=False, **kwargs):
        # method = kwargs.pop('local_and_nonlocal', 'global')
        mask = []
        k=0
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.local_g = dgl.batch([ex.local_g for ex in graph_list]).to(device)
        # bg.local_edges = torch.cat([ex.local_edges for ex in graph_list], dim=0).to(device)
        bg.local_mask = torch.cat([ex.graph.local_mask for ex in ex_list], dim=0).to(device)
        bg.global_g = dgl.batch([ex.global_g for ex in graph_list]).to(device)
        bg.global_edges = torch.cat([ex.global_edges for ex in graph_list], dim=0).to(device)
        bg.node_features = torch.cat([ex.node_features for ex in graph_list], dim=0).to(device)
        #print(bg.node_features[:5])
        for ex in enumerate(graph_list):
            len = ex[1].node_features.size(0)
            for i in range(len):
                mask.append(k)
            k += 1
        bg.mask = torch.tensor(mask, dtype=torch.long).to(device)
        if is_tool==False:
            bg.target_label = torch.cat([ex.target_label.unsqueeze(0) for ex in graph_list], dim=0).to(device)
        return bg
