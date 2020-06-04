import numpy as np
import scipy.sparse as sps



def sparse_to_tuple(sparse_mx):
        sparse_mx = sps.triu(sparse_mx)
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape