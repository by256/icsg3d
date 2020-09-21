"""
## Script for training the Cond-DFC-VAE
## Example:
## >> python3 train_vae.py --name heusler --samples 5000 --epochs 100 
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
import random
import re
import warnings

import numpy as np

import tensorflow as tf
from utils import data_split
from vae.data import VAEDataGenerator
from vae.lattice_vae import LatticeDFCVAE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # surpress tf warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", metavar="name", type=str, help="Name of data folder")
    parser.add_argument(
        "--samples",
        metavar="samples",
        type=int,
        help="Total number of training and validation samples",
        default=40000,
    )
    parser.add_argument(
        "--epochs",
        metavar="epochs",
        type=int,
        help="NUmber of epochs to train",
        default=50,
    )
    parser.add_argument(
        "--batch_size",
        metavar="batch_size",
        type=int,
        help="Batch size for training",
        default=20,
    )
    parser.add_argument(
        "--ncond",
        metavar="ncond",
        type=int,
        help="Number of condition bins",
        default=10,
    )
    parser.add_argument(
        "--nrot", metavar="nrot", type=int, help="Number of augmentations", default=10
    )
    parser.add_argument(
        "--cond",
        metavar="cond",
        type=str,
        help="Wether or not to condition the vae latent space",
        default='formation_energy_per_atom',
    )
    parser.add_argument(
        "--split",
        metavar="split",
        type=float,
        help="Train-test split fraction",
        default=0.8,
    )
    parser.add_argument(
        "--d",
        metavar="d",
        type=int,
        help="Dimension of density matrices (number of voxels)",
        default=32,
    )
    namespace = parser.parse_args()

    mode = namespace.name
    path = os.path.join("data", mode, "matrices")
    csv_path = os.path.join("data", mode, mode + ".csv")
    d = namespace.d
    input_shape = (d, d, d, 4)
    n = namespace.samples
    batch_size = namespace.batch_size
    epochs = namespace.epochs
    condition = namespace.cond
    weights_dir = os.path.join("saved_models", "vae", mode)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.join("output", "vae", mode), exist_ok=True)
    weights = os.path.join(weights_dir, "vae_weights_" + mode + ".best.hdf5")
    perceptual_model = os.path.join(
        "saved_models", "unet", mode, "unet_weights_" + mode + ".best.h5"
    )

    # Train-test split
    training_ids, validation_ids = data_split(
        path, n, frac=namespace.split, n_rot=namespace.nrot
    )
    # Make sure ids are unit of batch size
    if len(training_ids) % batch_size != 0:
        training_ids = training_ids[:-1 * int(len(training_ids) % batch_size)]
    if len(validation_ids) % batch_size != 0:
        validation_ids = validation_ids[:-1*int(len(validation_ids) % batch_size)]
    print(len(training_ids), len(validation_ids))

    # Create the VAE data generators
    training_generator = VAEDataGenerator(
        training_ids,
        data_path=path,
        property_csv=csv_path,
        batch_size=batch_size,
        n_channels=input_shape[-1],
        shuffle=True,
        n_bins=namespace.ncond,
        target=condition
    )
    validation_generator = VAEDataGenerator(
        validation_ids,
        data_path=path,
        property_csv=csv_path,
        batch_size=batch_size,
        n_channels=input_shape[-1],
        shuffle=True,
        n_bins=namespace.ncond,
        target=condition
    )

    # # Train
    lattice_vae = LatticeDFCVAE(
        perceptual_model=perceptual_model,
        cond_shape=namespace.ncond,
        output_dir=os.path.join("output", "vae", mode),
    )
    lattice_vae.train(
        training_generator, validation_generator, epochs=epochs, weights=weights
    )