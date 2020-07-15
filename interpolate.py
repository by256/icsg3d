"""ICSG3D/interpolate.py
Script for visualising interpolations in the latent space """

import argparse
import os
import random
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatgen as mg
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical
from matplotlib import rc

from sklearn.manifold import TSNE
from unet.unet import AtomUnet
from utils import to_lattice_params, to_pymatgen_structure, to_voxel_params
# warnings.filterwarnings('ignore')
from vae.lattice_vae import LatticeDFCVAE
from viz import plot_points_3d, viz
from watershed import watershed_clustering

font = {'family': 'serif'}
rc('font', **font)
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # surpress tf warnings
matplotlib.use('TkAgg')

def interpolate(a, b, cond, vae, num_interps=8, return_zs=False, max_alpha=1):
    """ Linearly Interpolate between two M's"""
    output = a
    _,_,z_a = vae.encoder.predict([a,cond])
    _,_,z_b = vae.encoder.predict([b,cond])
    vae.batch_size = num_interps

    z_a2b = z_b - z_a
    alpha = np.linspace(0, max_alpha, num_interps)
    z_interps = z_a + (alpha[:,np.newaxis] * z_a2b)
    M_interps = vae.decoder.predict([z_interps, np.tile(cond, (num_interps,1))])  # (num_interps, 32,32,32,4)
    output = np.concatenate([output, M_interps], axis=0)
    output = np.concatenate([output, b], axis=0)
    if return_zs:
        return output, np.concatenate([z_a, z_interps, z_b], axis=0)
    return output

def slerp(z_a, z_b, t):
    a = z_a/np.linalg.norm(z_a).reshape(len(z_a), 1)
    b = z_b/np.linalg.norm(z_b).reshape(len(z_b), 1)
    omega = np.arccos(np.einsum('ij,ij->i', a, b))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so*z_a + np.sin(t*omega)/so*z_b

def interpolate_slerp(a, b, cond, vae, num_interps=8, return_zs=False):
    """ Slerp interpolation between two M's"""
    output = a
    _,_,z_a = vae.encoder.predict([a,cond])
    _,_,z_b = vae.encoder.predict([b,cond])
    vae.batch_size = num_interps
    alpha = np.linspace(0, 1, num_interps)
    z_interps = np.array([slerp(z_a, z_b, t) for t in alpha]).reshape(num_interps, 256)
    M_interps = vae.decoder.predict([z_interps, np.tile(cond, (num_interps,1))])  # (num_interps, 32,32,32,4)
    output = np.concatenate([output, M_interps], axis=0)
    output = np.concatenate([output, b], axis=0)
    if return_zs:
        return output, np.concatenate([z_a, z_interps, z_b], axis=0)
    return output




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', metavar='name', type=str, help='Name of data folder')
    parser.add_argument('--ninterps', metavar='ninterps', type=int, help='Number of interpolations', default=8)
    parser.add_argument('--projection', metavar='projection', type=str, help='Dimensionality of visualisation', default=None)
    parser.add_argument('--enda', metavar='enda', type=str, help='End member of full interpolation', default='CeCrO3')
    parser.add_argument('--endb', metavar='endb', type=str, help='End member of full interpolation', default='YbCrO3')
    parser.add_argument('--ncond', metavar='ncond', type=int, help='Number of condition bins', default=10)
    parser.add_argument('--interpolate', metavar='interpolate', type=str, help='Type of interpolation', default='linear')
    namespace = parser.parse_args()

    conds = np.arange(namespace.ncond)
    if namespace.projection == 'None':
        projection = None
    else:
        projection = namespace.projection

    mode = namespace.name
    csv_path = os.path.join('data', mode, mode+'.csv')
    data_path = os.path.join('data', mode, 'matrices')
    target = 'formation_energy_per_atom'
    n_interps=namespace.ninterps
    os.makedirs(os.path.join('output/', 'interpolation'), exist_ok=True)

    vae_weights = os.path.join('saved_models', 'vae', mode, 'vae_weights_' + mode + '.best.hdf5')
    unet_weights = os.path.join('saved_models', 'unet', mode, 'unet_weights_' + mode + '.best.hdf5')
    unet_model = os.path.join('saved_models', 'unet', mode, 'unet_weights_' + mode + '.best.h5')

    df = pd.read_csv(csv_path)
    df['bin'] = pd.qcut(df[target], namespace.ncond, np.arange(namespace.ncond)).astype(int)

    vae = LatticeDFCVAE(perceptual_model=unet_model, cond_shape=namespace.ncond)
    vae._set_model(weights=vae_weights, batch_size=20)
    rows = []
    names = []
    for i in conds:
        print(i)
        cond = to_categorical(i, num_classes=namespace.ncond).reshape(1, namespace.ncond)
        ids = df[df['bin'] == i]['task_id'].values
        formulae = df[df['bin'] == i]['pretty_formula'].values
        # Pick a and b at random with the desired condition
        idxs = np.random.choice(np.arange(len(ids)), 2, replace=False)


        a_id = ids[idxs[0]]
        b_id = ids[idxs[1]]
        names.append((formulae[idxs[0]], formulae[idxs[1]]))

        # Load a and b
        Ma = np.load(os.path.join(data_path, 'density_matrices',  a_id + '_rot_2.npy')).reshape(1,32,32,32,1)
        Mb = np.load(os.path.join(data_path, 'density_matrices',  b_id + '_rot_4.npy')).reshape(1,32,32,32,1)

        Ca = np.load(os.path.join(data_path, 'coordinate_grids',  a_id + '_rot_2.npy')).reshape(1,32,32,32,3)
        Cb = np.load(os.path.join(data_path, 'coordinate_grids',  b_id + '_rot_4.npy')).reshape(1,32,32,32,3)

        Ma = np.concatenate([Ma, Ca], axis=-1)
        Mb = np.concatenate([Mb, Cb], axis=-1)
        if namespace.interpolate == 'linear':
            Ms = interpolate(Ma, Mb, cond, vae, num_interps=n_interps)
        elif namespace.interpolate == 'slerp':
            Ms = interpolate_slerp(Ma, Mb, cond, vae, num_interps=n_interps)
        rows.append(Ms)

    # Plot the rows
    fig, axes = plt.subplots(len(conds), n_interps+2, subplot_kw={'projection': projection}, figsize=(15,15))
    for i, row in enumerate(rows):
        for j in range(n_interps+2):
            if j == 0:
                axes[i][j].set_title(names[i][0], fontsize=12)
            if j == n_interps+1:
                axes[i][j].set_title(names[i][1], fontsize=12)
            if projection is None:
                axes[i][j].imshow(row[j,:,:,12,0])
            else:
                axes[i][j] = viz(row[j,:,:,:,0], ax=axes[i][j], show=False, resample_d=(15,15,15), alpha=0.15)
                axes[i][j].set_zticks([])
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.3)
    plt.savefig('output/interpolation/' + mode + '_rows.svg', format='svg')
    plt.show(block=True)