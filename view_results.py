""" ICSG3D/view_results.py
Script for viewing the results of each pipeline phase """

import argparse
import os
import re
import warnings
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymatgen as mg
import tensorflow as tf
from keras.utils import to_categorical
from scipy.spatial.distance import cdist, euclidean

from sklearn.metrics import mean_squared_error
from unet.data import UnetDataGenerator
from unet.unet import AtomUnet, f1_m
from utils import (coordinate_grid, create_crystal, data_split, density_matrix,
                   get_sites, to_lattice_params, to_pymatgen_structure)
from vae.data import VAEDataGenerator
from vae.lattice_vae import LatticeDFCVAE
from viz import animate, plot_points_3d, points, viz
from watershed import watershed_clustering

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # surpress tf warnings
matplotlib.use('TkAgg')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', metavar='name', type=str, help='Name of data folder')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('--nrot', metavar='nrot', type=int, help='nrot', default=10)
    parser.add_argument('--samples', metavar='samples', type=int, help='Number of samples', default=78750)
    parser.add_argument('--split', metavar='split', type=float, help='Train-test split fraction', default=0.8)
    namespace = parser.parse_args()

    batch_size = namespace.batch_size
    input_shape = (32,32,32,4)
    n = namespace.samples
    mode = namespace.name
    path = os.path.join('data', mode, 'matrices')
    cif_path = os.path.join('data', mode, 'cifs')
    csv_path = os.path.join('data', mode, mode+'.csv')
    vae_weights = os.path.join('saved_models', 'vae', mode, 'vae_weights_'+mode+'.best.hdf5')
    unet_weights = os.path.join('saved_models', 'unet', mode, 'unet_weights_'+mode+'.best.hdf5')
    perceptual_model = os.path.join('saved_models', 'unet', mode, 'unet_weights_' + mode + '.best.h5')

    training_ids, validation_ids = data_split(path, n, frac=namespace.split, n_rot=namespace.nrot)
    _generator = VAEDataGenerator(validation_ids, data_path=path, property_csv=csv_path, batch_size=batch_size, n_channels=input_shape[-1], shuffle=True, return_S=True)

    # LOAD VAE
    vae = LatticeDFCVAE(perceptual_model=perceptual_model)
    vae._set_model(batch_size=batch_size, weights=vae_weights)

    # LOAD UNET
    unet = AtomUnet(weights=unet_weights)
    for M, t in _generator:
        ids = _generator.list_IDs_temp
        print(ids[0])
        cond = t[0]
        S = np.argmax(t[1], axis=-1).reshape(batch_size, 32,32,32,1)
        S_b = t[2]
        c = np.load(path + '/coordinate_grids/' + ids[0]).reshape(1,32,32,32,3)
        Sg = np.concatenate([S, c], axis=-1)
        print(np.unique(S, return_counts=True))
        try:
            species, mus, R = watershed_clustering(M[:,:,:,:,0], S, S_b, return_ws=True, max_iters=5)
        except Exception:
            continue
        coords = np.concatenate([species.reshape((len(species), 1)), mus], axis=-1)

        M_recon = vae.model.predict([M, cond])
        mse = mean_squared_error(M.reshape(32*32*32, 4), M_recon.reshape(32*32*32,4))
        c_recon = M_recon[:,:,:,:,1:].reshape(1,32,32,32,3)
        S_recon, S_b_recon = unet.model.predict(M_recon)  # (1,32,32,32,95)
        f1 = f1_m(to_categorical(S, num_classes=95), S_recon)
        S_recon = np.argmax(S_recon, axis=-1).reshape(1,32,32,32,1)
        print(np.unique(S_recon, return_counts=True))
        S_b_recon= np.where(S_b_recon >= 0.8, 1, 0)
        try:
            species_pred, mus_pred, R_pred = watershed_clustering(M_recon[:,:,:,:,0], S_recon, S_b_recon, max_iters=5, return_ws=True)
        except Exception:
            continue
        coords_pred = np.concatenate([species_pred.reshape((len(species_pred), 1)), mus_pred], axis=-1)

        fig = plt.figure()
        ax = fig.add_subplot(241, projection='3d')
        ax = viz(M[0,:,:,:,0], ax=ax, show=False, resample_d=(10,10,10))
        ax.set_title('M True')

        ax = fig.add_subplot(242, projection='3d')
        ax = plot_points_3d(S.squeeze(), ax=ax)
        ax.set_title('S True')

        ax = fig.add_subplot(243, projection='3d')
        ax = plot_points_3d(R.squeeze(), ax=ax)
        ax.set_title('WS True')

        ax = fig.add_subplot(244, projection='3d')
        ax = points(coords, ax=ax, show=False)
        ax.set_title('Coords True')

        ax = fig.add_subplot(245, projection='3d')
        ax = viz(M_recon[0,:,:,:,0], ax=ax, show=False, resample_d=(10,10,10))
        ax.set_title('M Pred, MSE = %.3f' % mse)

        ax = fig.add_subplot(246, projection='3d')
        ax = plot_points_3d(S_recon.squeeze(), ax=ax)
        ax.set_title('S pred F1 = %.3f' % f1)

        ax = fig.add_subplot(247, projection='3d')
        ax = plot_points_3d(R_pred.squeeze(), ax=ax)
        ax.set_title('WS Pred')

        ax = fig.add_subplot(248, projection='3d')
        ax = points(coords_pred, ax=ax, show=False)
        ax.set_title('Coords Pred')

        plt.show(block=True)
        inp = input('...')
        if inp == 'n':
            break
