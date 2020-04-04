"""
ICSG3D/create_matrices.py
Script for creating the density, species, coordinates and lattice vector matrices
-- uses mpi4py
"""
import argparse
import os
import sys

import numpy as np

from func_timeout import FunctionTimedOut
from mpi4py import MPI
from utils import (coordinate_grid, create_crystal, density_matrix, get_sites,
                   random_rotation_3d)
from viz import plot_points_3d

if __name__ == '__main__':
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', metavar='name', type=str, help='Name of data folder')
    parser.add_argument('--d', metavar='d', type=int, help='Dimensionality of the matrices', default=32)
    parser.add_argument('--nrot', metavar='nrot', type=int, help='NUmber of rotations to apply', default=10)
    parser.add_argument('--label_frac', metavar='label_frac', type=float, help='Fraction of ionic radius to label in species matrices', default=1.0)
    parser.add_argument('--sigma_frac', metavar='sigma_frac', type=float, help='Fraction of ionic radius to use as Gaussian width', default=1.0)
    parser.add_argument('--eps_frac', metavar='eps_frac', type=float, help='Fraction of unit cell to add along each dimension', default=0.25)
    parser.add_argument('--max_sites', metavar='max_sites', type=int, help='Maximum number of sites in the unit cell', default=40)
    namespace = parser.parse_args()
    
    mode = namespace.name
    d = namespace.d
    dims = (d, d, d)
    n_rot = namespace.nrot
    label_frac = namespace.label_frac
    eps_frac=namespace.eps_frac
    sigma_frac=namespace.sigma_frac
    max_sites = namespace.max_sites

    data_path = os.path.join('data', mode, 'cifs')
    csv_path = os.path.join('data', mode, mode+'.csv')
    sdir = os.path.join('data', mode, 'matrices')

    if rank == 0:
        os.makedirs(sdir, exist_ok=True)
        os.makedirs(os.path.join(sdir, 'density_matrices'), exist_ok=True)
        os.makedirs(os.path.join(sdir, 'species_matrices'), exist_ok=True)
        os.makedirs(os.path.join(sdir, 'lattice_vectors'), exist_ok=True)
        os.makedirs(os.path.join(sdir, 'coordinate_grids'), exist_ok=True)
        
    MPI.COMM_WORLD.Barrier()

    for root, dirs, files in os.walk(data_path):
        for i, fname in enumerate(files):
            if i % size == rank:
                if not fname.endswith('.cif'):
                    continue
                try:
                    path = os.path.join(root, fname)
                    try:
                        crystal = create_crystal(path, primitive=False)
                    except FunctionTimedOut:
                        continue
                    
                    N, z, r = get_sites(crystal)
                    if len(N) > max_sites:
                        continue
                    Nm = ((N <= 1.0) & (N >= 0))
                    if not np.all(Nm):
                        print(fname, N)
                        sys.stdout.flush()
                        sys.exit()
                        break
                    (a,b,c) = crystal.lattice.abc
                    lattice_vector = np.array([a, b, c, crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma])
                    N = np.multiply(N, lattice_vector[:3])
                    p = coordinate_grid(lattice_vector, eps_frac=eps_frac, dim=d)
                    if any(np.isnan(r)):
                        continue
                    try:
                        M, S = density_matrix(N, z, l=[a,b,c], sigma=sigma_frac*r, dims=dims, label_frac=label_frac, eps_frac=eps_frac)
                    except FunctionTimedOut:
                        print(i, " timeout")
                        continue
                    np.save(os.path.join(sdir, 'density_matrices', fname.strip('.cif')), M)
                    np.save(os.path.join(sdir, 'species_matrices', fname.strip('.cif')), S)
                    np.save(os.path.join(sdir, 'lattice_vectors', fname.strip('.cif')), lattice_vector)
                    np.save(os.path.join(sdir, 'coordinate_grids', fname.strip('.cif')), p)
                    # Apply k random rotations to each
                    for k in range(n_rot):
                        m_rot_k, s_rot_k, p_rot_k = random_rotation_3d(M, S, p)
                        np.save(os.path.join(sdir, 'density_matrices', fname.strip('.cif')+'_rot_' + str(k)), m_rot_k)
                        np.save(os.path.join(sdir, 'species_matrices', fname.strip('.cif')+'_rot_' + str(k)), s_rot_k)
                        np.save(os.path.join(sdir, 'lattice_vectors', fname.strip('.cif')+'_rot_' + str(k)), lattice_vector)
                        np.save(os.path.join(sdir, 'coordinate_grids', fname.strip('.cif')+'_rot_' + str(k)), p_rot_k)
                    
                    print(rank, i)
                    sys.stdout.flush()
                except Exception as e:
                    print(e)
                    sys.stdout.flush()
                    break
