import tensorflow as tf

from tensorflow.keras import Model

from algo.VGAE import VGAEModel


class DynVGAE(Model):
    def __init__(self, cell_num, layer_1_in, layer_1_out, layer_2_out, dropout):
        super(DynVGAE, self).__init__()
        self.cell_num = cell_num
        self.vgae_cells = [VGAEModel(layer_1_in, layer_1_out, layer_2_out, dropout) for _ in range(self.cell_num)]

    def call(self, features, adjs):
        zs = []
        z_means = []
        z_stds = []
        reconstructions = []
        for i, vgae_cell in enumerate(self.vgae_cells):
            z, z_mean, z_std, reconstruction = vgae_cell(features[i], adjs[i])
            zs.append(z)
            z_means.append(z_mean)
            z_stds.append(z_std)
            reconstructions.append(reconstruction)
        return zs, z_means, z_stds, reconstructions

