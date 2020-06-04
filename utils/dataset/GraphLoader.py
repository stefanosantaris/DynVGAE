import networkx as nx
import numpy as np
import scipy.sparse as sps

class GraphLoader():
    def __init__(self, path):
        super(GraphLoader, self).__init__()
        self.path = path

    def load_graph(self, file):
        return nx.read_weighted_edgelist(self.path + "edges" + str(file) +".csv", delimiter=',', nodetype=int,encoding='utf-8')

    def read_adjacency(self, file, max_id):
        G = self.load_graph(file)
        node_list = list(G.nodes())
        node_list.sort()
        if node_list[-1] > max_id:
            max_id = node_list[-1]

        adj = np.zeros((max_id + 1, max_id + 1))
        edge_list = G.edges(data=True)
        for edge in edge_list:
            src = edge[0]
            dst = edge[1]
            weight = edge[2]['weight']
            adj[src][dst] = weight
            adj[dst][src] = weight
        return sps.csr_matrix(adj, dtype=float)