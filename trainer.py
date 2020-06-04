import tensorflow as tf

from algo.DynVGAE import DynVGAE
from utils.dataset.GraphLoader import GraphLoader
from utils.utils import sparse_to_tuple
import scipy.sparse as sps
import numpy as np
import random
import time

class Trainer():
    def __init__(self, exp_params):
        super(Trainer, self).__init__()
        self.graph_loader = GraphLoader(exp_params['path'] + "/" + exp_params['extract_folder'] +"/")

    def prepare_test_adj(self, input_graph, ground_truth_adj):
        coords, values, shape = sparse_to_tuple(input_graph)
        ground_truth_adj = (ground_truth_adj[:input_graph.shape[0], :input_graph.shape[1]]).todense()
        for coord in coords:
            ground_truth_adj[coord[0], coord[1]] = 0.
            ground_truth_adj[coord[1], coord[0]] = 0.
        return sps.triu(sps.csr_matrix(ground_truth_adj, dtype=float))

    def normalize(self, adj):
        adj_with_diag = adj + sps.identity(adj.shape[0], dtype=np.float32).tocsr()
        rowsum = np.array(adj_with_diag.sum(1))
        degree_mat_inv_sqrt = sps.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_with_diag.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo().astype(np.float32)
        return adj_normalized


    def construct_dataset(self, graph, window_size, negative_sample):
        start_graph = max(0, graph - window_size + 1)
        max_id = 0
        for i in range(start_graph, graph + 1):
            adj = self.graph_loader.read_adjacency(i, max_id)
            max_id = adj.shape[0] - 1

        train_adj_sps = []
        total_train_edges = np.zeros((max_id + 1, max_id + 1))
        for i in range(start_graph, graph + 1):
            adj = self.graph_loader.read_adjacency(i, max_id)
            tmp_train_adj_dense = adj.todense()
            tmp_train_adj_dense = np.where(tmp_train_adj_dense > 0.2, tmp_train_adj_dense, 0)
            tmp_train_adj_sparse = sps.csr_matrix(tmp_train_adj_dense)
            coords, values, shape = sparse_to_tuple(tmp_train_adj_sparse)
            for coord in coords:
                total_train_edges[coord[0], coord[1]] = 1
            train_adj_sps.append(tmp_train_adj_sparse)


        # Construct a full matrix with ones to generate negative sample tuples
        train_ns = np.ones_like(total_train_edges) - total_train_edges - sps.identity(total_train_edges.shape[0])
        ns_coord, ns_values, ns_shape = sparse_to_tuple(train_ns)

        train_adj_norm = []
        features = []
        train_adj_labels = []
        train_adj_inds = []
        features_tuples = sparse_to_tuple(sps.identity(adj.shape[0], dtype=np.float32, format='coo'))
        for i, adj in enumerate(train_adj_sps):
            adj_norm_coord, adj_norm_values, adj_norm_shape = sparse_to_tuple(self.normalize(adj))
            train_adj_norm.append(tf.SparseTensor(indices=adj_norm_coord,
                                                  values=np.array(adj_norm_values, dtype='float32'),
                                                  dense_shape=[adj_norm_shape[0], adj_norm_shape[1]]))

            features.append(tf.SparseTensor(indices=features_tuples[0], values=features_tuples[1],
                                            dense_shape=[features_tuples[2][0], features_tuples[2][1]]))

            tmp_train_adj_dense = adj.todense()
            train_coord, train_values, train_shape = sparse_to_tuple(adj)
            tmp_train_adj_ind = np.zeros_like(tmp_train_adj_dense)

            sequence = [i for i in range(len(ns_coord))]
            random_coords = set(random.sample(sequence, negative_sample * len(train_coord)))

            for coord in train_coord:
                tmp_train_adj_ind[coord[0], coord[1]] = 1

            for coord in random_coords:
                tmp_train_adj_ind[ns_coord[coord][0], ns_coord[coord][1]] = 1

            nnz_ind = np.nonzero(tmp_train_adj_ind)
            tmp_train_label_val = tmp_train_adj_dense[nnz_ind]
            train_adj_label_tensor = tf.convert_to_tensor(tmp_train_label_val, dtype=tf.float32)
            train_adj_labels.append(train_adj_label_tensor)

            ind_list = []
            for i in range(len(nnz_ind[0])):
                ind_list.append([nnz_ind[0][i], nnz_ind[1][i]])
            train_adj_inds.append(tf.convert_to_tensor(ind_list, dtype=tf.int32))



        test_adj_dense = self.prepare_test_adj(sps.csr_matrix(total_train_edges), self.graph_loader.read_adjacency(graph + 1, max_id)).todense()
        test_adj_high = np.where(test_adj_dense > 0.2, test_adj_dense, 0)
        test_adj_ind = np.where(test_adj_high > 0., 1, 0)
        nnz_ind = np.nonzero(test_adj_ind)

        ind_list = []
        for i in range(len(nnz_ind[0])):
            ind_list.append([nnz_ind[0][i], nnz_ind[1][i]])

        test_adj = tf.convert_to_tensor(test_adj_high[nnz_ind], dtype=tf.float32)
        test_adj_ind = tf.convert_to_tensor(ind_list, dtype=tf.int32)


        return train_adj_norm, train_adj_labels, train_adj_inds, features, test_adj, test_adj_ind

    def count_parameters(self,model):
        return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])


    def get_edge_embeddings(self, embeddings, indices):
        src_embeddings = tf.gather(embeddings, indices[:,0])
        dst_embeddings = tf.gather(embeddings, indices[:,1])
        return tf.multiply(src_embeddings, dst_embeddings)

    def evaluate_model(self, emb_size, train_embeddings, train_values, test_embeddings, test_values):
        evaluation_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(emb_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
        evaluation_model.compile(loss=tf.keras.losses.MSE, optimizer='adam')

        evaluation_model.fit(train_embeddings, train_values,
                             epochs=10, verbose=0, batch_size=512)

        test_res = evaluation_model(test_embeddings)
        m = tf.keras.metrics.RootMeanSquaredError()
        m.update_state(test_values, test_res)
        rmse_score = m.result().numpy()

        m = tf.keras.metrics.MeanAbsoluteError()
        m.update_state(test_values, test_res)
        mae_score = m.result().numpy()

        return mae_score, rmse_score


    def train_model(self, args):
        num_exp = args.num_exp
        start_graph = args.start_graph
        end_graph = args.end_graph
        dropout = args.dropout
        negative_sample = args.ns
        emb = args.emb
        window_size = args.window
        learning_rate = args.learning_rate

        results = {}
        print("Start training")
        for graph in range(start_graph, end_graph + 1):
            results[graph] = {'num_params': 0, 'mae': 0., 'rmse': 0.}
            mae = []
            rmse = []
            number_of_params = []
            print("Construct Dataset")
            train_adj_norm, train_adj_label, train_adj_ind, features, test_adj, test_adj_ind = self.construct_dataset(graph, window_size, negative_sample)
            print("Start experimentation")
            for i in range(num_exp):
                print("Experiment {} for GRAPH {}".format(i, graph))
                device = "/GPU:0"
                if args.cuda == 0:
                    device = "/CPU:0"
                with tf.device(device):
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    model = DynVGAE(cell_num=len(train_adj_norm), layer_1_in=train_adj_norm[0].shape[0], layer_1_out=2 * emb, layer_2_out = emb, dropout=dropout)
                    for epoch in range(100):
                        with tf.GradientTape() as tape:
                            z, z_mean, z_std, reconstruction = model(features, train_adj_norm)

                            total_loss = 0
                            previous_kls = []
                            for k in range(len(z)):
                                reconstruct_val = tf.gather_nd(reconstruction[k], train_adj_ind[k])
                                m = tf.keras.metrics.RootMeanSquaredError()
                                m.update_state(reconstruct_val, train_adj_label[k])
                                reconstruction_loss = m.result().numpy()

                                # KL Divergence
                                kl = (0.5 / train_adj_norm[k].shape[0]) * tf.reduce_mean(
                                    tf.reduce_sum(1 + 2 * z_std[k] - tf.square(z_mean[k]) - tf.square(tf.exp(z_std[k])), 1))

                                previous_kls.append(kl)
                                final_kl = tf.reduce_mean(previous_kls[-len(train_adj_norm):])

                                total_loss += reconstruction_loss - final_kl

                        grads = tape.gradient(total_loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    number_of_params.append(self.count_parameters(model))

                    z, z_mean,z_std, reconstruction = model(features, train_adj_norm)
                    train_edge_embeddings = tf.convert_to_tensor(self.get_edge_embeddings(z[-1], train_adj_ind[-1]), dtype=tf.float32)
                    train_values = tf.reshape(train_adj_label[-1],[train_adj_label[-1].shape[1], 1])
                    test_edge_embeddings = tf.convert_to_tensor(self.get_edge_embeddings(z[-1], test_adj_ind), dtype=tf.float32)
                    test_values = test_adj
                    mae_score, rmse_score = self.evaluate_model(emb, train_edge_embeddings, train_values, test_edge_embeddings, test_values)
                    mae.append(mae_score)
                    rmse.append(rmse_score)

                    tf.keras.backend.clear_session()

            del train_adj_norm, train_adj_label, train_adj_ind, features, test_adj, test_adj_ind


            results[graph]['num_params'] = np.mean(number_of_params)
            results[graph]['mae'] = np.mean(mae)
            results[graph]['rmse'] = np.mean(rmse)
            print(
                "Graph {} : N_PARAMS {} : MAE {} : RMSE {}".format(
                    graph,
                    results[graph]['num_params'],
                    results[graph]['mae'],
                    results[graph]['rmse']
                ))
        return results

