"""
## General Utility functions
--------------------------------------------------
## Author: Callum J. Court.
## Email: cc889@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Callum Court & Batuhan Yildirim 2020, ICSG3D
-------------------------------------------------
"""


import json
import os
import random
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial.distance import cdist, euclidean
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans

import pymatgen as mg
from func_timeout import FunctionTimedOut, func_set_timeout, func_timeout
from pymatgen.io.cif import CifParser
from pymatgen.transformations.standard_transformations import (
    AutoOxiStateDecorationTransformation,
    OrderDisorderedStructureTransformation)

warnings.filterwarnings("ignore")


def data_split(path, n=None, frac=0.80, n_rot=10, shuffle=True, seed=28):
    """ Train-test split the matrices data"""

    training_ids = []
    validation_ids = []
    data_ids = sorted(
        [x for x in os.listdir(path + "/density_matrices") if x.endswith(".npy")]
    )
    data_ids_plain = [x for x in data_ids if not "_rot_" in x][:n]
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(data_ids_plain)
    training_ids_plain = data_ids_plain[: int(frac * len(data_ids_plain))]
    validation_ids_plain = data_ids_plain[int(frac * len(data_ids_plain)) :]
    assert list(set(training_ids_plain) & set(validation_ids_plain)) == []
    for i in training_ids_plain:
        training_ids.append(i)
        for r in range(n_rot):
            training_ids.append(i.strip(".npy") + "_rot_" + str(r) + ".npy")
    for i in validation_ids_plain:
        validation_ids.append(i)
        for r in range(n_rot):
            validation_ids.append(i.strip(".npy") + "_rot_" + str(r) + ".npy")
    assert list(set(training_ids) & set(validation_ids)) == []
    return training_ids, validation_ids


def get_sites(crystal):
    """ Returns the coords, ionic radii and atomic numbers of sites in a pymatgen structure """
    N = np.zeros((len(crystal.sites), 3))  # nsites x 3
    z = np.zeros(len(N))  # nsites x 1
    r = np.zeros(len(N))  # ionic radii
    for i, site in enumerate(crystal.sites):
        if hasattr(site.specie, "element"):
            site_z = site.specie.element.number
            ir = np.mean([v for v in site.specie.element.ionic_radii.values()])
            if np.isnan(ir):
                ir = 1.20
        else:
            site_z = site.specie.number
            ir = np.mean([v for v in site.specie.ionic_radii.values()])
            if np.isnan(ir):
                ir = 1.20

        N[i, :] = site.frac_coords
        z[i] = site_z
        r[i] = ir
    trans_vec = np.min(N, axis=0)
    N = N - trans_vec
    return N, z, r


def coordinate_grid(l, dim=32, eps_frac=0.25):
    """ Returns a dxdxdx3 coordinate grid given a lattice vector """
    x = np.linspace(0, l[0] + 2 * eps_frac * l[0], dim + 1)[:-1]
    y = np.linspace(0, l[1] + 2 * eps_frac * l[1], dim + 1)[:-1]
    z = np.linspace(0, l[2] + 2 * eps_frac * l[2], dim + 1)[:-1]
    pos_grid = list(product(x, y, z))
    return np.array(pos_grid).reshape((dim, dim, dim, 3))


@func_set_timeout(180)
def density_matrix(
    N, z, l, dims=(32, 32, 32), sigma=0.5, dist=euclidean, label_frac=1.0, eps_frac=0.25
):
    """ Compute density maps and species matrices """
    a = l[0]
    b = l[1]
    c = l[2]
    dx = (a + (2 * a * eps_frac)) / dims[0]
    dy = (b + (2 * b * eps_frac)) / dims[1]
    dz = (c + (2 * c * eps_frac)) / dims[2]
    dv = np.array([dx, dy, dz])
    xc = np.linspace(-a * eps_frac, a + (a * eps_frac), dims[0] + 1)[:-1]
    yc = np.linspace(-b * eps_frac, b + (b * eps_frac), dims[1] + 1)[:-1]
    zc = np.linspace(-c * eps_frac, c + (c * eps_frac), dims[2] + 1)[:-1]

    # Voxel coordinates (bottom left)
    V_bls = np.array(list(product(xc, yc, zc)))
    # Voxel centres
    V = V_bls + dv / 2
    # Pairwise distances between sites and voxel centres
    D = cdist(V, N, metric=euclidean)  # Nvoxels x n sites
    S = np.zeros(D.shape)
    # For each voxel get the closest site
    for v, voxel in enumerate(D):
        for s, _site in enumerate(voxel):
            if D[v, s] < sigma[s] * label_frac:
                # If this voxel is already assigned choose the closest
                if sum(S[v, :]) > 0:
                    min_idx = np.argmin(D[v, :])
                    new = np.zeros(len(voxel))
                    new[min_idx] = 1
                    S[v, :] = new
                    continue
                else:
                    S[v, s] = 1

    S = np.dot(S, z)
    S = S.reshape(dims)
    D = D ** 2
    z = z / (sigma ** 3)
    D = np.exp(-1 * D / (2 * sigma ** 2))
    k = np.dot(D, z)
    a = 1.0 / ((2 * np.pi) ** (1.5))
    m = a * k

    # reshape
    M = m.reshape((dims[0], dims[1], dims[2]))
    return M, S

@func_set_timeout(120)
def create_crystal(cifpath, primitive=False):
    """ Convert cif to pymatgen struture """
    structure = CifParser(cifpath).get_structures(primitive=primitive)[0]
    if structure.is_ordered:
        return structure
    order_transformer = OrderDisorderedStructureTransformation()
    oxid_transformer = AutoOxiStateDecorationTransformation()
    a = oxid_transformer.apply_transformation(structure)
    b = order_transformer.apply_transformation(a)
    return b


def to_lattice_params(p, eps_frac=0.25, d=32, axis=(-3, -2, -1)):
    """ Convert a density matrix to lattice params """
    batch = len(p)
    ap = np.max(p[:, :, :, :, 0], axis=axis) - np.min(p[:, :, :, :, 0], axis=axis)
    ap = ap / (1 + 2 * eps_frac)
    ap = ap / (1 - 1.0 / d)
    bp = np.max(p[:, :, :, :, 1], axis=axis) - np.min(p[:, :, :, :, 1], axis=axis)
    bp = bp / (1 + 2 * eps_frac)
    bp = bp / (1 - 1.0 / d)
    cp = np.max(p[:, :, :, :, 2], axis=axis) - np.min(p[:, :, :, :, 2], axis=axis)
    cp = cp / (1 + 2 * eps_frac)
    cp = cp / (1 - 1.0 / d)
    ap -= ap / d
    bp -= bp / d
    cp -= cp / d
    lp = np.concatenate(
        [ap.reshape(batch, 1), bp.reshape(batch, 1), cp.reshape(batch, 1)], axis=-1
    )
    return lp


def to_voxel_params(lp, eps=0.25, d=32):
    """ Determine voxel size from lattice params """
    batch = len(lp)
    dx = (lp[:, 0] + (2 * lp[:, 0] * eps)) / d  # (batch, 1)
    dy = (lp[:, 1] + (2 * lp[:, 1] * eps)) / d
    dz = (lp[:, 2] + (2 * lp[:, 2] * eps)) / d
    dv = np.concatenate(
        [dx.reshape(batch, 1), dy.reshape(batch, 1), dz.reshape(batch, 1)], axis=-1
    )
    return dv


def random_rotation_3d(M, S, p, rot_angle=90, nrotations=3):
    """Randomly rotate an image nrotations times around a random axis each time.

    Arguments:
    rot_angle: `float`. The rotation angle.

    Returns:
    batch of rotated 3D images
    """
    axes_choices = [(0, 1), (0, 2), (1, 2)]
    rotations = [axes_choices[x] for x in np.random.choice(3, 3)]

    M_rot = M
    S_rot = S
    p_rot = p

    for i in range(nrotations):
        M_rot = scipy.ndimage.interpolation.rotate(
            M_rot, rot_angle, mode="nearest", axes=rotations[i], reshape=False
        )
        S_rot = scipy.ndimage.interpolation.rotate(
            S_rot, rot_angle, mode="nearest", axes=rotations[i], reshape=False
        )
        p_rot = scipy.ndimage.interpolation.rotate(
            p_rot, rot_angle, mode="nearest", axes=rotations[i], reshape=False
        )
    S_rot = np.abs(np.rint(S_rot))
    p_rot[np.abs(p_rot) < 1e-14] = 0
    assert np.array_equal(np.unique(S_rot), np.unique(S))
    return M_rot, S_rot, p_rot


def to_pymatgen_structure(lattice_params, coords, alpha=90, beta=90, gamma=90):
    """ Convert lattice params and coords to pymatgen """
    species = [c[0] for c in coords]
    positions = np.array([c[1:] for c in coords])

    # Create a lattice to place the sites
    lattice = mg.Lattice.from_parameters(*lattice_params, alpha=90, beta=90, gamma=90)

    # Create the structure
    structure = mg.Structure(lattice, species, positions, coords_are_cartesian=True)
    new_lattice = mg.Lattice.from_parameters(
        *lattice_params, alpha=alpha, beta=beta, gamma=gamma
    )
    structure.lattice = new_lattice
    return structure
