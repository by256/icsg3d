"""
## Script for training the UNet Semantic Segmentation Network
## Example:
## >> python3 train_unet.py --name heusler --samples 5000 --epochs 100 
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

import numpy as np

import tensorflow as tf
from unet.data import UnetDataGenerator
from unet.get_weights import get_weights
from unet.unet import AtomUnet
from utils import data_split
from vae.lattice_vae import LatticeDFCVAE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # surpress tf warnings

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", metavar="name", type=str, help="Name of data folder")
    parser.add_argument(
        "--samples",
        metavar="samples",
        type=int,
        help="Total number of training and validation samples",
        default=20000,
    )
    parser.add_argument(
        "--d",
        metavar="d",
        type=int,
        help="Dimension of density matrices (number of voxels)",
        default=32,
    )
    parser.add_argument(
        "--epochs",
        metavar="epochs",
        type=int,
        help="Number of epochs to train",
        default=50,
    )
    parser.add_argument(
        "--lr", metavar="lr", type=float, help="Learning rate", default=3e-6
    )
    parser.add_argument(
        "--batch_size",
        metavar="batch_size",
        type=int,
        help="Batch size for training",
        default=10,
    )
    parser.add_argument(
        "--nrot", metavar="nrot", type=int, help="Batch size for training", default=10
    )
    parser.add_argument(
        "--nclasses",
        metavar="nclasses",
        type=int,
        help="Number of segmentation classes",
        default=95,
    )
    parser.add_argument(
        "--split",
        metavar="split",
        type=float,
        help="Train-test split fraction",
        default=0.8,
    )
    namespace = parser.parse_args()

    mode = namespace.name
    d = namespace.d
    path = os.path.join("data", mode, "matrices")
    input_shape = (d, d, d, 4)
    samples = namespace.samples
    epochs = namespace.epochs
    weights_dir = os.path.join("saved_models", "unet", mode)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.join("output/unet", mode), exist_ok=True)
    weights = os.path.join(weights_dir, "unet_weights_" + mode + ".best.hdf5")
    lr = namespace.lr
    batch_size = namespace.batch_size

    # Split the data
    training_ids, validation_ids = data_split(
        path, samples, frac=namespace.split, n_rot=namespace.nrot
    )
    training_generator = UnetDataGenerator(
        training_ids,
        data_path=path,
        batch_size=batch_size,
        n_channels=input_shape[-1],
        shuffle=True,
    )
    validation_generator = UnetDataGenerator(
        validation_ids,
        data_path=path,
        batch_size=batch_size,
        n_channels=input_shape[-1],
        shuffle=True,
    )

    # Calculate class weights from the training set, or pre-load
    try:
        class_weights = np.load(weights_dir + "/class_weights.npy")
    except Exception:
        class_weights = get_weights(path, training_ids, namespace.nclasses)
        class_weights[0] = 0.0
        np.save(weights_dir + "/class_weights.npy", class_weights)

    unet = AtomUnet(
        num_classes=namespace.nclasses,
        class_weights=class_weights,
        input_shape=input_shape,
        weights=weights,
        lr=lr,
    )

    # Train
    unet.train_generator(
        training_generator,
        validation_generator,
        epochs=epochs,
        output_dir=os.path.join("output", "unet", mode),
    )
    # Save the final model as .h5
    unet.save_(weights, os.path.splitext(weights)[0] + '.h5')
