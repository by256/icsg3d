"""
## Script for generating new crystal structures around a specific base compound (or a random one)
## Example:
## >> python3 generate.py --name heusler --
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
import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from matplotlib import cm

import pymatgen as mg
from cgcnn.cgcnn import CGCNN
from cgcnn.utils import evaluate_cgcnn_from_cif
from unet.unet import AtomUnet
from utils import to_lattice_params, to_pymatgen_structure, to_voxel_params
from vae.data import VAEDataGenerator
from vae.lattice_vae import LatticeDFCVAE
from viz import (
    animate,
    expand_coordinates,
    explode,
    plot_points_3d,
    resize,
    tsne_latent,
    viz,
)
from watershed import watershed_clustering

matplotlib.use("TkAgg")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", metavar="name", type=str, help="Name of data folder")
    parser.add_argument(
        "--base", metavar="base", type=str, help="Base Compound", default="LaFeO3"
    )
    parser.add_argument(
        "--batch_size", metavar="batch_size", type=int, help="Batch size", default=100
    )
    parser.add_argument(
        "--nsamples",
        metavar="nsamples",
        type=int,
        help="Number of samples",
        default=100,
    )
    parser.add_argument(
        "--var", metavar="var", type=float, help="Variance of sampling", default=0.5
    )
    parser.add_argument(
        "--eps_frac",
        metavar="eps_frac",
        type=float,
        help="Eps of lattice vector",
        default=0.25,
    )
    parser.add_argument(
        "--clus_iters",
        metavar="clus_iters",
        type=int,
        help="Iterations of Clustering",
        default=5,
    )
    parser.add_argument("--alpha", metavar="alpha", type=int, help="alpha", default=90)
    parser.add_argument("--beta", metavar="beta", type=int, help="beta", default=90)
    parser.add_argument("--gamma", metavar="gamma", type=int, help="gamma", default=90)
    parser.add_argument(
        "--target", metavar="target", type=str, default="formation_energy_per_atom"
    )
    parser.add_argument(
        "--ncond",
        metavar="ncond",
        type=int,
        help="Number of condition bins",
        default=10,
    )

    namespace = parser.parse_args()

    mode = namespace.name
    nsamples = namespace.nsamples
    batch_size = namespace.batch_size

    path = os.path.join("data", mode, "matrices")
    variance = namespace.var
    eps = namespace.eps_frac
    clustering_max_iters = namespace.clus_iters

    df = pd.read_csv(os.path.join("data", mode, mode + ".csv"))
    df["interval"] = pd.qcut(
        df[namespace.target], namespace.ncond, np.arange(namespace.ncond)
    )
    training_names = df["pretty_formula"].values
    n_samples = namespace.nsamples
    batch_size = namespace.batch_size
    d = namespace.d
    input_shape = (d, d, d, 4)

    properties = [
        "formation_energy_per_atom",
        "bulk_modulus",
        "dielectric_constant",
        "eij_max",
        "refractive_index",
        "shear_modulus",
        "energy_per_atom",
        "bandgap",
    ]

    base_compound = namespace.base
    if base_compound.startswith("mp-"):
        base_formula = df[df["task_id"] == base_compound]["pretty_formula"].values[0]
    else:
        base_formula = df[df["pretty_formula"] == base_compound][
            "pretty_formula"
        ].values[0]
        base_compound = df[df["pretty_formula"] == base_compound]["task_id"].values[0]
    base_target_value = df[df["task_id"] == base_compound][namespace.target].values[0]

    vae_weights = os.path.join(
        "saved_models", "vae", mode, "vae_weights_" + mode + ".best.hdf5"
    )
    unet_weights = os.path.join(
        "saved_models", "unet", mode, "unet_weights_" + mode + ".best.hdf5"
    )
    perceptual_model = os.path.join(
        "saved_models", "unet", mode, "unet_weights_" + mode + ".best.h5"
    )

    alpha = namespace.alpha
    beta = namespace.beta
    gamma = namespace.gamma

    # Output dirs
    out_dir = os.path.join(
        "output",
        "results",
        base_formula + "_" + str(datetime.now()) + "_v=" + str(variance),
    )
    os.makedirs(os.path.join(out_dir, "cifs"))
    os.makedirs(os.path.join(out_dir, "densities"))
    os.makedirs(os.path.join(out_dir, "species"))

    # Load VAE
    vae = LatticeDFCVAE(perceptual_model=perceptual_model)
    vae._set_model(vae_weights, batch_size=batch_size)

    # Load Unet
    unet = AtomUnet(weights=unet_weights)

    # Load CGCNN
    cgcnn = CGCNN(batch_size=1)

    # Load the M, C, cond of the base compound
    M_base = np.load(
        os.path.join(path, "density_matrices", base_compound + ".npy")
    ).reshape(1, 32, 32, 32, 1)
    S_base = np.load(
        os.path.join(path, "species_matrices", base_compound + ".npy")
    ).reshape(1, 32, 32, 32, 1)
    C_base = np.load(
        os.path.join(path, "coordinate_grids", base_compound + ".npy")
    ).reshape(1, 32, 32, 32, 3)
    M_base = np.concatenate([M_base, C_base], axis=-1)
    cond_val = df[df["task_id"] == base_compound]["interval"].values
    cond = to_categorical(cond_val, num_classes=10).reshape(1, 10)

    # Encode with the vae
    z_mu_base, z_logvar_base, z_base = vae.encoder.predict([M_base, cond])

    results = []
    formulae = []

    # Draw n_samples from around the base
    for batch in range(int(n_samples / batch_size)):
        print("Batch", batch)
        z_samples = np.random.normal(z_mu_base, variance, size=(batch_size, 256))

        # Decode the samples
        cond_tensor = np.tile(cond, (batch_size, 1))
        M_prime_samples = vae.decoder.predict([z_samples, cond_tensor])

        # Get the lattice params
        C_prime_samples = M_prime_samples[:, :, :, :, 1:].reshape(
            batch_size, 32, 32, 32, 3
        )
        l_prime_samples = to_lattice_params(C_prime_samples)

        # Get the voxel params
        dv_pred = to_voxel_params(l_prime_samples)

        # Run through unet
        S_prime_samples, S_b_prime_samples = unet.model.predict(M_prime_samples)
        S_prime_samples = np.argmax(S_prime_samples, axis=-1).reshape(
            batch_size, 32, 32, 32, 1
        )
        S_b_prime_samples[S_b_prime_samples >= 0.8] = 1.0
        S_b_prime_samples[S_b_prime_samples < 0.8] = 0.0

        # segment
        for i in range(batch_size):
            # Get coordinates
            try:
                species_sample, mu_samples = watershed_clustering(
                    M_prime_samples[i, :, :, :, 0],
                    S_prime_samples[i],
                    S_b_prime_samples[i],
                    max_iters=clustering_max_iters,
                )
                mu_samples = (
                    mu_samples * dv_pred[i]
                    - (eps * l_prime_samples[i])
                    + (dv_pred[i] / 2.0)
                )
                coords = np.concatenate(
                    [species_sample.reshape((len(species_sample), 1)), mu_samples],
                    axis=-1,
                )
            except Exception:
                print("Failed")
                continue

            # Write to cifs
            structure = to_pymatgen_structure(
                l_prime_samples[i], coords, alpha, beta, gamma
            )
            comp = mg.Composition(structure.formula)
            formula = structure.formula

            if formula in formulae:
                formula_count = formula + "_" + str(formulae.count(formula))
            else:
                formula_count = formula

            if (
                not structure.is_valid()
                or not comp.anonymized_formula in desired_structure
            ):
                continue

            filename = os.path.join(out_dir, "cifs", formula_count + ".cif")
            writer = mg.io.cif.CifWriter(structure)
            writer.write_file(filename)

            # Predict properties
            rd = {
                "formula": formula,
                "id": formula_count,
                "training": 0,
                "target": base_target_value,
                "anon_formula": comp.anonymized_formula,
            }
            if comp.reduced_formula in training_names:
                rd["training"] = 1

            property_predictions = {}
            for prop in properties:
                property_pred = evaluate_cgcnn_from_cif(
                    cgcnn, filename, weights=[prop], batch_size=1
                )
                property_predictions[prop] = property_pred[0][0][0]
                rd[prop] = property_pred[0][0][0]

            rd["target_diff_pct"] = np.abs(
                (rd[namespace.target] - base_target_value) / base_target_value
            )
            rd["electronegativity"] = comp.average_electroneg
            rd["charge balanced"] = (
                1 if comp.oxi_state_guesses(all_oxi_states=True) else 0
            )
            rd["cif"] = filename

            results.append(rd)
            formulae.append(structure.formula)

            np.save(
                os.path.join(out_dir, "densities", formula_count), M_prime_samples[i]
            )
            np.save(os.path.join(out_dir, "species", formula_count), S_prime_samples[i])

            print(
                batch * batch_size + i,
                formula,
                structure.is_valid(),
                mg.Composition(structure.formula).anonymized_formula,
                rd[namespace.target],
            )
            # Write results to json
            with open(os.path.join(out_dir, "results.json"), "a+") as wf:
                json.dump(str(rd), wf)

    df = pd.DataFrame(results)
    df = df.sort_values(by=namespace.target)
    df.to_csv(os.path.join(out_dir, "results.csv"))
