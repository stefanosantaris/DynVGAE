import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

class GraphConvolution(Layer):


    def __init__(self, input_dim, output_dim, dropout, act=tf.nn.relu, sparse=False):
        super(GraphConvolution, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform()
        self.W = tf.Variable(initializer(shape=[input_dim, output_dim]), trainable=True, dtype=tf.float32)
        self.b = tf.Variable(tf.zeros(output_dim), trainable=True, dtype=tf.float32)
        self.dropout = dropout
        self.act = act
        self.sparse = sparse

    # def weight_variable_glorot(self, input_dim, output_dim):
    #     """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    #     initialization.
    #     """
    #     init_range = np.sqrt(6.0 / (input_dim + output_dim))
    #     initial = tf.random.uniform([input_dim, output_dim], minval=-init_range,
    #                                 maxval=init_range, dtype=tf.float32)
    #     return tf.Variable(initial)

    def call(self, features, adj):

        if self.sparse:
            # Add dropout on features for the first convolution. It is sparse.
            x = tf.sparse.sparse_dense_matmul(features, self.W)
        else:
            x = tf.nn.dropout(features, rate=self.dropout)
            x = tf.matmul(x, self.W)

        x = tf.sparse.sparse_dense_matmul(adj, x)
        return self.act(x + self.b)



class Encoder(Layer):
    def __init__(self, layer_1_in, layer_1_out, layer_2_out, dropout):
        super(Encoder,self).__init__()
        self.gc1 = GraphConvolution(layer_1_in, layer_1_out, dropout, act=tf.nn.relu, sparse=True)
        self.z_mean = GraphConvolution(layer_1_out, layer_2_out, dropout, act=lambda x: x, sparse=False)
        self.z_std = GraphConvolution(layer_1_out, layer_2_out, dropout, act=lambda x: x, sparse=False)

    def call(self, features, adj):
        hidden = self.gc1(features, adj)
        embeddings = self.z_mean(hidden, adj)
        std = self.z_std(hidden, adj)
        return embeddings, std


class Decoder(Layer):
    def __init__(self, act=tf.nn.relu):
        super(Decoder,self).__init__()
        self.act = act

    def call(self, z):
        x = tf.transpose(z)
        x = tf.matmul(z, x)
        return self.act(x)




class VGAEModel(Model):
    def __init__(self, layer_1_in, layer_1_out, layer_2_out, dropout):
        super(VGAEModel, self).__init__()
        self.layer_1_in = layer_1_in
        self.layer_1_out = layer_1_out
        self.layer_2_out = layer_2_out
        self.dropout = dropout
        self.encoder = Encoder(layer_1_in, layer_1_out, layer_2_out, dropout)
        self.decoder = Decoder(act = tf.nn.sigmoid)

    def call(self, features, adj):
        z_mean, z_std = self.encoder(features, adj)
        z = z_mean + tf.random.normal([adj.shape[0], self.layer_2_out])  * tf.exp(z_std)
        reconstruction = self.decoder(z)
        return z, z_mean, z_std, reconstruction

