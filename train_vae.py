"""
ICSG3D/train_vae.py
Script for traiing the VAE architecture
"""
import argparse
import os
import random
import re
import warnings

import numpy as np
import tensorflow as tf

from utils import data_split
from vae.data import VAEDataGenerator
# warnings.filterwarnings('ignore')
from vae.lattice_vae import LatticeDFCVAE

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # surpress tf warnings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', metavar='name', type=str, help='Name of data folder')
    parser.add_argument('--samples', metavar='samples', type=int, help='Total number of training and validation samples', default=40000)
    parser.add_argument('--epochs', metavar='epochs', type=int, help='NUmber of epochs to train', default=50)
    parser.add_argument('--batch_size', metavar='batch_size', type=int, help='Batch size for training', default=20)
    parser.add_argument('--ncond', metavar='ncond', type=int, help='Number of condition bins', default=10)
    parser.add_argument('--nrot', metavar='nrot', type=int, help='Number of augmentations', default=10)
    parser.add_argument('--cond', metavar='cond', type=int, help='Wether or not to condition the vae latent space', default=1)
    parser.add_argument('--split', metavar='split', type=float, help='Train-test split fraction', default=0.8)
    namespace = parser.parse_args()
    
    mode = namespace.name
    path = os.path.join('data', mode, 'matrices')
    csv_path = os.path.join('data', mode, mode+'.csv')
    input_shape=(32,32,32,4)
    n = namespace.samples
    batch_size=namespace.batch_size
    epochs = namespace.epochs
    condition = namespace.cond
    weights_dir = os.path.join('saved_models', 'vae', mode)
    os.makedirs(weights_dir, exist_ok=True)
    weights = os.path.join(weights_dir, 'vae_weights_' + mode + '.best.hdf5')
    perceptual_model = os.path.join('saved_models', 'unet', mode, 'unet_weights_' + mode + '.best.h5')

    training_ids, validation_ids = data_split(path, n, frac=namespace.split, n_rot=namespace.nrot)
    training_ids = training_ids[:-1*int(len(training_ids)%batch_size)]
    validation_ids = validation_ids[:-1*int(len(validation_ids)%batch_size)]
    print(len(training_ids), len(validation_ids))

    training_generator = VAEDataGenerator(training_ids, data_path=path, property_csv=csv_path, batch_size=batch_size, n_channels=input_shape[-1], shuffle=True, n_bins=namespace.ncond)
    validation_generator = VAEDataGenerator(validation_ids, data_path=path, property_csv=csv_path, batch_size=batch_size, n_channels=input_shape[-1], shuffle=True, n_bins=namespace.ncond)


    # # Train
    lattice_vae = LatticeDFCVAE(condition=condition, perceptual_model=perceptual_model, cond_shape=namespace.ncond)
    lattice_vae.train(training_generator, validation_generator, epochs=epochs, weights=weights)
