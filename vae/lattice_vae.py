"""
## Conditional Deep Feature Consistent VAE model
--------------------------------------------------
## Author: Callum J. Court.
## Email: cc889@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Callum Court & Batuhan Yildirim 2020, ICSG3D
-------------------------------------------------
"""



import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.datasets import mnist
from keras.layers import (BatchNormalization, Concatenate, Conv2D, Conv3D,
                          Conv3DTranspose, Dense, Dropout, Flatten, Input,
                          Lambda, LeakyReLU, MaxPool3D, ReLU, Reshape,
                          UpSampling3D)
from keras.losses import binary_crossentropy, mse
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import Sequence, plot_model, to_categorical
from sklearn.manifold import TSNE

import tensorflow as tf
from unet.unet import custom_objects
from viz import imscatter, viz

matplotlib.use('Agg')



def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class LatticeDFCVAE():
    """
    Conditional Variational Auto-encoder with a deep feature consistent model

    params:
    input_shape: Dimensions of the density matrix inputs, default (32,32,32,4)
    kernel_size: kernel dim for Conv3D layers
    pool_size: max pool dimensions
    filters: Number of filters at each conv layer
    latent_dim: size of bottleneck
    beta: Weighting of KLD loss term
    alpha: Weightin of DFC loss term
    perceptual_model: Path to pre-trained Unet model (.h5)
    pm_layers: Which layers to use of unet for DFC calcs
    pm_layer_Weights: Weights of each DFC layer
    cond_shape: Dimension of condition vector
    custom_objects: Losses and metrics for the unet

    """

    def __init__(self,
                 input_shape=(32,32,32,4),
                 kernel_size=(3,3,3),
                 pool_size=(2,2,2),
                 filters=[16,32,64,128],
                 latent_dim=256,
                 beta=3e-4,
                 alpha=0.5,
                 optimizer=Adam(5e-4),
                 perceptual_model='saved_models/unet.h5',
                 pm_layers=['re_lu_2', 're_lu_4', 're_lu_6', 're_lu_8'],
                 pm_layer_weights=[1.0, 1.0, 1.0, 1.0],
                 cond_shape=10,
                 custom_objects=custom_objects):
        self.input_shape=input_shape
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.filters = filters
        self.latent_dim=latent_dim
        self.channels=self.input_shape[-1]
        self.optimizer = optimizer
        self.beta = beta
        self.alpha = alpha
        self.batch_size = None
        self.cond_shape = cond_shape
        self.losses = []

        self.pm = load_model(perceptual_model, custom_objects=custom_objects)
        self.pm_layers = pm_layers
        self.pm_layer_weights = pm_layer_weights

        self.metrics = [self.perceptual_loss, self.mse_loss, self.kld_loss]
        self.metric_names = ['Loss', 'PM', 'MSE', 'KLD']

    def _set_model(self, weights=None, batch_size=20):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        M_input = Input(batch_shape=(self.batch_size,) + self.input_shape)
        cond_input = Input(batch_shape=(self.batch_size,self.cond_shape))
        z_mean, z_log_var, z = self.encoder([M_input, cond_input])
        reconstructed = self.decoder([z, cond_input])
        self.model = Model(inputs=[M_input, cond_input], outputs=reconstructed)

        self.z = z
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        self.model.compile(optimizer=self.optimizer, loss=self._vae_dfc_loss(self.alpha, self.beta), metrics=self.metrics)
        # self.decoder.compile(optimizer=self.optimizer, loss='mse')
        self.batch_size = batch_size

        if weights and os.path.exists(weights):
            self.model.load_weights(weights)
            self.filepath = weights
        elif weights and not os.path.exists(weights):
            self.filepath = weights
        else:
            self.filepath = 'saved_models/lattice_dfc_vae_weights.best.hdf5'
        
        # print("Model setup complete...")
        return

    def build_encoder(self):
        """
        Encoder model
        """

        M_input = Input(batch_shape=(self.batch_size,) + self.input_shape)
        cond_input = Input(batch_shape=(self.batch_size, self.cond_shape))
        cond = Reshape((1,1,1, self.cond_shape))(cond_input)
        cond = Lambda(K.tile, arguments={'n': self.input_shape})(cond)
        x = Concatenate()([M_input, cond])

        for i in range(len(self.filters)):
            f = self.filters[i]
            x = Conv3D(filters=f,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = MaxPool3D(pool_size=self.pool_size)(x)
            
        x = Conv3D(filters=4,kernel_size=self.kernel_size, padding='same')(x)
        x = LeakyReLU()(x)

        #generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(self.latent_dim, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model([M_input, cond_input], [z_mean, z_log_var, z], name='encoder')
        return encoder

    def build_decoder(self):
        """
        Decoder model
        """

        latent_inputs = Input(batch_shape=(self.batch_size, self.latent_dim), name='decoder_input')  # (None, 256)
        cond_inputs = Input(batch_shape=(self.batch_size, self.cond_shape))
        #concatenate the condition
        z_cond = Concatenate()([latent_inputs, cond_inputs])
        x = Dense(self.latent_dim)(z_cond)
        x = Reshape((4,4,4,4))(x)

        for i in range(len(self.filters)):
            f = self.filters[::-1][i]
            x = Conv3D(filters=f,
                        kernel_size=self.kernel_size,
                        padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if i < len(self.filters) - 1:
                x = UpSampling3D(self.pool_size)(x)
        
        outputs = Conv3D(filters=self.channels,
                        kernel_size=self.kernel_size,
                        padding='same',
                        name='decoder_output')(x)
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)

        # instantiate decoder model
        decoder = Model([latent_inputs, cond_inputs], outputs, name='decoder')
        return decoder
    
    def mse_loss(self, inputs, outputs):
        return mse(K.flatten(inputs), K.flatten(outputs))
    
    def kld_loss(self, y_true=None, y_pred=None):
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss

    def _vae_dfc_loss(self, alpha, beta):
        """ Combined loss function """

        alpha = K.variable(alpha)
        beta = K.variable(beta)
        def loss(x, t_decoded):
            '''Total loss for the DFC VAE'''
            rs_loss = self.mse_loss(x, t_decoded)
            kl_loss = self.kld_loss()
            pm_loss = self.perceptual_loss(x, t_decoded)

            return K.mean(rs_loss + (alpha * pm_loss) + (beta * kl_loss))
        return loss
    
    def perceptual_loss(self, y_true, y_pred):
        """ Perceptual model loss """

        outputs = [self.pm.get_layer(l).output for l in self.pm_layers]
        model = Model(self.pm.input, outputs)
        h1_list = model(y_true)
        h2_list = model(y_pred)
        rc_loss = 0.0
        
        for h1, h2, weight in zip(h1_list, h2_list, self.pm_layer_weights):
            h1 = K.batch_flatten(h1)
            h2 = K.batch_flatten(h2)
            rc_loss += weight * K.mean(K.square(h1 - h2), axis=-1)
        return rc_loss
    
    def train(self, train_gen, val_gen, epochs, weights=None):
        best_loss = np.inf
        self.train_batch_size = train_gen.batch_size
        self.val_batch_size = val_gen.batch_size
        self.batch_size = self.train_batch_size
        
        self.num_epochs = epochs
        train_steps_per_epoch = int(len(train_gen.list_IDs)/self.train_batch_size)
        val_steps_per_epoch = int(len(val_gen.list_IDs)/self.val_batch_size)
        print("Data size %d,    batch_size %d    steps per epoch %d" % (len(train_gen.list_IDs), self.train_batch_size, train_steps_per_epoch))
        
        self._set_model(weights)
        self.losses = np.empty((self.num_epochs, 2))
        for e in range(self.num_epochs):
            print('Epoch %s:' % e)
            t0 = time.time()
            # Training
            # Storing losses for computing mean
            train_metrics = []
            for batch_idx in range(train_steps_per_epoch):
                train_batch, train_cond = train_gen[batch_idx]
                batch_metrics = self.model.train_on_batch([train_batch, train_cond], train_batch)

                train_metrics.append(np.array(list(batch_metrics)))
            train_metrics = np.mean(train_metrics, axis=0)
            epoch_train_loss = train_metrics[0]
            # Validation
            val_metrics = []
            for batch_idx in range(val_steps_per_epoch):
                val_batch, val_cond = val_gen[batch_idx]
                batch_metrics = self.model.test_on_batch([val_batch, val_cond], val_batch)
                val_metrics.append(np.array(list(batch_metrics)))
            val_metrics = np.mean(val_metrics, axis=0)
            t1 = time.time()
            epoch_str = 'Time: %.3f s   ' %(t1 - t0)
            for m in range(len(self.metrics) + 1):
                epoch_str += 'Train %s: %.3f    ' % (self.metric_names[m], train_metrics[m])
            for m in range(len(self.metrics) + 1):
                epoch_str += 'Val %s: %.3f    ' % (self.metric_names[m], val_metrics[m])
            print(epoch_str)
            
            epoch_val_loss = val_metrics[0]
            self.losses[e,] = [epoch_train_loss, epoch_val_loss]
            if e % 10 == 0:
                self.plot_losses(epoch=e, name='output/vae/vae_losses.png')
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                self.plot_reconstructions(val_gen, epoch=e, name='output/vae/vae_reconstructions_best.png')
                self.plot_samples(self.batch_size, epoch=e, name='output/vae/vae_samples_best.png')
                self.plot_kde(val_gen, epoch=e, name='output/vae/kde_best.png')
                print("Saving Model")
                self.model.save_weights(self.filepath)
        self.model.load_weights(self.filepath)
        self.model.save(os.path.splitext(self.filepath)[0] + '.h5')
        print("Model saved")
    
    def save_(self, weights, model='saved_models/vae.h5'):
        self.model.load_weights(weights)
        self.model.save(model)
        return
    
    def sample_vae(self, n_samples, cond=None, var=1.0):
        if cond is None:
            cond = np.random.randint(low=0, high=self.cond_shape, size=n_samples)

        cond_tensor = to_categorical(cond, num_classes=self.cond_shape)
        cond_tensor = np.tile(cond_tensor, (n_samples, 1))
        z_sample = np.random.normal(0, var, size=(n_samples, self.latent_dim))
        output = self.decoder.predict([z_sample, cond_tensor])
        return z_sample, output

    def plot_samples(self, n_samples=20, epoch=0, name=None):
        _, samples = self.sample_vae(n_samples)
        fig, axes = plt.subplots(10, 2)
        ax = 0
        for i in range(0, n_samples, 2):  # 0, 2, 4, 6, 8
            axes[ax][0].imshow(samples[i,:,:,16,0])
            axes[ax][1].imshow(samples[i+1,:,:,16,0])
            axes[ax][0].set_xticks([])
            axes[ax][0].set_yticks([])
            axes[ax][1].set_xticks([])
            axes[ax][1].set_yticks([])
            ax += 1
            if ax == 10:
                break
        if name is None:
            name = 'output/vae/samples_epoch_%d.png' % epoch
        plt.tight_layout()
        plt.savefig(name)
        plt.close()
        return
    
    def plot_reconstructions(self, val_gen, epoch, name=None):
        fig, axes = plt.subplots(10, 2)
        ax = 0
        for M, cond in val_gen:
            if self.condition:
                recon = self.model.predict([M, cond])
            else:
                recon = self.model.predict(M)
            axes[ax][0].imshow(M[0, :,:,16,0])
            axes[ax][1].imshow(recon[0,:,:,16,0])
            axes[ax][0].set_xticks([])
            axes[ax][0].set_yticks([])
            axes[ax][1].set_xticks([])
            axes[ax][1].set_yticks([])

            ax += 1
            if ax == 10:
                break
        if name is None:
            name = 'output/vae/reconstructions_epoch_%d.png' % epoch
        plt.tight_layout()
        plt.savefig(name)
        plt.close()
        return
    
    def plot_kde(self, val_gen, epoch, name=None, maxz=1000):
        z = []
        # Real samples
        for M, p in val_gen:
            if self.condition:
                _,_,z_m = self.encoder.predict([M, p])
            else:
                _,_,z_m = self.encoder.predict(M)
            for iz in range(len(z_m)):
                z.append(z_m[iz])
            if len(z) >= maxz:
                break
        z = np.array(z)
        x = np.linspace(-3, 3, 50)

        fig, ax = plt.subplots(1,1)
        for i in range(self.latent_dim):
            density = stats.gaussian_kde(z[:, i])
            ax.plot(x, density(x))
        plt.xlabel('$x$')
        plt.ylabel('Density')
        plt.show(block=True)
        plt.savefig(name, format='svg')
        plt.savefig(name, format='png')
        plt.close()

        return
    
    def plot_losses(self, epoch, name='output/vae/loss.png'):
        fig, ax = plt.subplots(1,1)
        ax.plot(self.losses[:epoch+1,0], label='Train Loss')
        ax.plot(self.losses[:epoch+1,1], label='Val loss')
        ax.set_xlabel('Epoch #')
        ax.set_ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(name)
        plt.close()
        return
