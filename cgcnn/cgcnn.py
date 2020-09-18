"""
## CGCNN model and crystal graph conv keras layer
--------------------------------------------------
## Author: Batuhan Yildirim
## Email: by256@cam.ac.uk
## Version: 1.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Callum Court & Batuhan Yildirim 2020, ICSG3D
-------------------------------------------------
"""

import tensorflow as tf
import keras.backend as K
from keras import activations
from keras.models import Model
from keras.activations import softplus
from keras.layers import Layer, Input, Dense


class CrystalGraphConv(Layer):
    """
    Crystal graph convolution Keras layer 
    https://journals-aps-org.ezp.lib.cam.ac.uk/prl/abstract/10.1103/PhysRevLett.120.145301
    """
    def __init__(self, atom_fea_len, nbr_fea_len, **kwargs):
        """
        Parameters
        ----------
        atom_fea_len : int
            Number of atom (node features).
        nbr_fea_len : int
            Number of bond (edge features).
        """
        super(CrystalGraphConv, self).__init__(**kwargs)
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
    
    def build(self, input_shape):
        self.gc_W = self.add_weight(name='graph_conv_weight', 
                                      shape=(2*self.atom_fea_len + self.nbr_fea_len, 2*self.atom_fea_len),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.gc_bias = self.add_weight(name='graph_conv_bias', 
                                      shape=(2*self.atom_fea_len, ),
                                      initializer='zeros',
                                      trainable=True)
        self.gamma_1 = self.add_weight(name='gamma_1', 
                                      shape=(2*self.atom_fea_len, ),
                                      initializer='ones',
                                      trainable=True)
        self.beta_1 = self.add_weight(name='beta_1', 
                                      shape=(2*self.atom_fea_len, ),
                                      initializer='zeros',
                                      trainable=True)
        self.gamma_2 = self.add_weight(name='gamma_2', 
                                      shape=(self.atom_fea_len, ),
                                      initializer='ones',
                                      trainable=True)
        self.beta_2 = self.add_weight(name='beta_2', 
                                      shape=(self.atom_fea_len, ),
                                      initializer='zeros',
                                      trainable=True)
        super(CrystalGraphConv, self).build(input_shape)

    def call(self, x):
        atom_fea, nbr_fea, nbr_fea_idx, mask = x
        _, N, M = nbr_fea_idx.shape
        atom_nbr_fea = tf.gather(atom_fea, indices=nbr_fea_idx, axis=1, batch_dims=1)        
        atom_fea_expanded = tf.tile(tf.expand_dims(atom_fea, axis=2), [1, 1, M, 1])
        total_nbr_fea = tf.concat([atom_fea_expanded, atom_nbr_fea, nbr_fea], axis=3)
        total_gated_fea = K.dot(total_nbr_fea, self.gc_W) + self.gc_bias
        total_gated_fea = total_gated_fea * K.cast(mask, tf.float32)

        # batch norm 1
        total_gated_fea = K.reshape(total_gated_fea, (-1, 2*self.atom_fea_len))
        mask_stacked_1 = K.reshape(mask, (-1, 2*self.atom_fea_len))

        mu_1 = tf.reduce_sum(total_gated_fea) / tf.math.count_nonzero(total_gated_fea, dtype=tf.float32)
        diff_squared_1 = (total_gated_fea - mu_1)**2 * K.cast(mask_stacked_1, tf.float32)
        var_1 = K.sum(diff_squared_1) / tf.math.count_nonzero(total_gated_fea, dtype=tf.float32)

        total_gated_fea = K.batch_normalization(total_gated_fea, mu_1, var_1, self.beta_1, self.gamma_1, epsilon=1e-5)
        total_gated_fea = K.reshape(total_gated_fea, (-1, N, M, 2*self.atom_fea_len))
        total_gated_fea = total_gated_fea * K.cast(mask, tf.float32)
        
        nbr_filter, nbr_core = tf.split(total_gated_fea, 2, axis=3)
        nbr_filter = K.sigmoid(nbr_filter)
        nbr_core = K.softplus(nbr_core)
        nbr_summed = K.sum(nbr_filter * nbr_core, axis=2) * K.cast(mask[:, :, 0, :self.atom_fea_len], tf.float32)

        # batch norm 2
        mu_2 = K.sum(nbr_summed) / tf.math.count_nonzero(nbr_summed, dtype=tf.float32)
        diff_squared_2 = (nbr_summed - mu_2)**2 * K.cast(mask[:, :, 0, :nbr_summed.shape[-1]], tf.float32)
        var_2 = K.sum(diff_squared_2) / tf.math.count_nonzero(diff_squared_2, dtype=tf.float32)

        nbr_summed = K.batch_normalization(nbr_summed, mu_2, var_2, self.beta_2, self.gamma_2, epsilon=1e-5)
        nbr_summed = nbr_summed * K.cast(mask[:, :, 0, :nbr_summed.shape[-1]], tf.float32)

        return K.softplus(atom_fea + nbr_summed) * K.cast(mask[:, :, 0, :self.atom_fea_len], tf.float32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.atom_fea_len)


class MaxPooling(Layer):
    """
    Global max pooling layer. 
    Computes the node-wise maximum over the node feature matrix of a graph.
    """
    def __init__(self, activation=None, **kwargs):
        self.activation = activations.get(activation)
        super(MaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaxPooling, self).build(input_shape)

    def call(self, x):
        return self.activation(tf.reduce_max(x, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class MeanPooling(Layer):
    """
    Global average pooling layer. 
    Computes the node-wise average over the node feature matrix of a graph.
    """
    def __init__(self, activation=None, **kwargs):
        self.activation = activations.get(activation)
        super(MeanPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MeanPooling, self).build(input_shape)

    def call(self, x):
        pooled = tf.reduce_sum(x, axis=1) / tf.expand_dims(tf.math.count_nonzero(tf.reduce_sum(x, axis=2), axis=1, dtype=tf.float32), axis=1)
        return self.activation(pooled)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def CGCNN(batch_size):
    """
    CGCNN Keras model
    """
    atom_features = Input(batch_shape=(batch_size, 50, 93), name='atom_input')
    bond_features = Input(batch_shape=(batch_size, 50, 12, 41), name='bond_input')
    atom_neighbour_idxs = Input(batch_shape=(batch_size, 50, 12), name='atom_n_input', dtype='int32')
    masks = Input(batch_shape=(batch_size, 50, 12, 128), name='masks_input', dtype='int32')

    x = Dense(64)(atom_features)
    x = CrystalGraphConv(64, 41)([x, bond_features, atom_neighbour_idxs, masks])
    x = MeanPooling(activation='softplus')(x)
    x = Dense(128, activation='softplus')(x)
    out = Dense(1)(x)

    return Model(inputs=[atom_features, bond_features, atom_neighbour_idxs, masks], outputs=out)