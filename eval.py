"""
ICSG3D/eval.py
Script for evaluating atomic coordinates and lattice params """

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from unet.unet import AtomUnet
from utils import (create_crystal, data_split, get_sites, to_lattice_params,
                   to_voxel_params)
from vae.data import VAEDataGenerator
from vae.lattice_vae import LatticeDFCVAE
from watershed import watershed_clustering

font = {'family': 'serif'}
rc('font', **font)
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')

def emd(x,y):
    dist = cdist(x, y)
    assign = linear_sum_assignment(dist)
    return dist[assign].sum() / min(len(x), len(y))


if __name__ == '__main__':

    """
    for each cif in validation set
    - pass through vae and unet
    - run watershed
    - calculate coords and n sites
    - compare to original
    - plot N_true vs N pred
    - plot hist of emd

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', metavar='name', type=str, help='Name of data folder')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, help='Batch size', default=10)
    parser.add_argument('--samples', metavar='samples', type=int, help='Number of samples', default=78750)
    parser.add_argument('--eps_frac', metavar='eps_frac', type=float, help='Eps of lattice vector', default=0.25)
    parser.add_argument('--ncond', metavar='ncond', type=int, help='Number of condition bins', default=10)
    parser.add_argument('--clus_iters', metavar='clus_iters', type=int, help='Number of iterations for watershed clustering', default=5)
    parser.add_argument('--split', metavar='split', type=float, help='Train-test split fraction', default=0.8)
    namespace = parser.parse_args()

    mode = namespace.name
    ncond = namespace.ncond
    data_path = os.path.join('data', mode, 'matrices')
    cif_path = os.path.join('data', mode, 'cifs')
    csv_path = os.path.join('data', mode, mode+'.csv')
    input_shape = (32,32,32,4)
    n = namespace.samples
    batch_size=namespace.batch_size
    eps = namespace.eps_frac
    vae_weights = os.path.join('saved_models', 'vae', mode, 'vae_weights_'+mode+'.best.hdf5')
    unet_weights = os.path.join('saved_models', 'unet', mode, 'unet_weights_'+mode+'.best.hdf5')
    perceptual_model = os.path.join('saved_models', 'unet', mode, 'unet_weights_' + mode + '.best.h5')
    clustering_max_iters = namespace.clus_iters

    os.makedirs(os.path.join('output', 'eval', mode), exist_ok=True)

    # Split the data
    training_ids, validation_ids = data_split(data_path, n, frac=namespace.split, n_rot=0)
    validation_generator = VAEDataGenerator(validation_ids, data_path=data_path, property_csv=csv_path, batch_size=batch_size, n_channels=input_shape[-1], shuffle=False, n_bins=ncond)

    vae = LatticeDFCVAE(perceptual_model=perceptual_model, cond_shape=ncond)
    vae._set_model(weights=vae_weights, batch_size=batch_size)
    unet = AtomUnet(weights=unet_weights)

    true_num_atoms = []
    pred_num_atoms = []
    true_species = []
    pred_species = []
    true_lc = []
    pred_lc = []
    true_coords = []
    pred_coords = []
    emds = []

    c = 0
    for M, cond in validation_generator:
        M_prime = vae.model.predict([M, cond])
        coords_prime = M_prime[:,:,:,:,1:]
        S_prime, S_b_prime = unet.model.predict(M_prime)
        S_prime = np.argmax(S_prime, axis=-1).reshape(batch_size, 32,32,32, 1)
        S_b_prime[S_b_prime >= 0.8] = 1.0
        S_b_prime[S_b_prime < 0.8] = 0.0
        S_prime_coords = np.concatenate([S_prime, coords_prime], axis=-1)

        l_pred = to_lattice_params(coords_prime)
        dv_pred = to_voxel_params(l_pred)

        ids = validation_generator.list_IDs_temp
        for i, S_prime_i in enumerate(S_prime_coords):
            print(ids[i])
            # True data
            true_id = ids[i]
            crystal = create_crystal(os.path.join(cif_path, re.split('_|\.', true_id)[0] + '.cif'), primitive=False)
            N, z, r = get_sites(crystal)
            lpt = [crystal.lattice.a, crystal.lattice.b, crystal.lattice.c]
            N = np.multiply(N, lpt[:3])
            dist = np.linalg.norm(N, ord=2, axis=1)
            N = N[np.argsort(dist)]

            # Predicted
            try:
                species, mu = watershed_clustering(M_prime[i,:,:,:,0], S_prime[i], S_b_prime[i])
            except Exception:
                print(ids[i], 'failed')
                continue
            

            for s in N:
                true_coords.append(s)
            true_lc.append(lpt)
            true_num_atoms.append(len(N))
            true_species.append(np.unique(z))
            
            pred_lc.append(l_pred[i])
            lpp = eps*l_pred[i, :3].reshape(1,3)
            mu = mu*dv_pred[i] - (lpp) + (dv_pred[i]/2.)
            dist = np.linalg.norm(mu, ord=2, axis=1)
            mu = mu[np.argsort(dist)]

            dist = emd(mu, N)
            emds.append(dist)

            # sort pred coords by dist from 0
            pred_num_atoms.append(len(species))
            pred_species.append(np.unique(species))

            c += 1
    
    true_num_atoms = np.array(true_num_atoms)
    pred_num_atoms = np.array(pred_num_atoms)
    true_lc = np.array(true_lc)
    pred_lc = np.array(pred_lc)

    print("\nMEAN EMD: ", np.mean(emds))
    print("\nMEAN DAtoms: ", np.mean(np.abs(true_num_atoms - pred_num_atoms)))
    
    plt.figure()
    plt.hist(emds, bins=50, color='tab:cyan')
    plt.axvline(x=np.mean(emds), linestyle='--', color='r', label='Mean = %.3f' % np.mean(emds))
    plt.xlabel('EMD (Angstrom)')
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.savefig('output/eval/' + mode + '/emd.svg', format='svg')
    plt.close()

    plt.figure()
    plt.hist(np.abs(true_lc - pred_lc)[:,0], bins=50, color='tab:cyan')
    plt.axvline(x=np.mean(np.abs(true_lc - pred_lc)[:,0]), linestyle='--', color='tab:red', label="Mean = %.3f" % np.mean(np.abs(true_lc - pred_lc)[:,0]))
    plt.xlabel('$|a_{true}$ - $a_{pred}|$ (Angstrom)')
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.savefig('output/eval/' + mode + '/lattice_a.svg', format='svg')
    plt.close()

    plt.figure()
    plt.hist(np.abs(true_lc - pred_lc)[:,1], bins=50, color='tab:cyan')
    plt.axvline(x=np.mean(np.abs(true_lc - pred_lc)[:,1]), linestyle='--', color='tab:red', label='Mean = %.3f' % np.mean(np.abs(true_lc - pred_lc)[:,1]))
    plt.xlabel('$|b_{true}$ - $b_{pred}|$ (Angstrom)')
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.savefig('output/eval/' + mode + '/lattice_b.svg', format='svg')
    plt.close()

    plt.figure()
    plt.hist(np.abs(true_lc - pred_lc)[:,2], bins=50, color='tab:cyan')
    plt.axvline(x=np.mean(np.abs(true_lc - pred_lc)[:,2]), linestyle='--', color='tab:red', label='Mean = %.3f' % np.mean(np.abs(true_lc - pred_lc)[:,2]))
    plt.xlabel('$|c_{true}$ - $c_{pred}|$ (Angstrom)')
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.savefig('output/eval/' + mode + '/lattice_c.svg', format='svg')
    plt.close()

    plt.figure()
    plt.hist(np.abs(true_num_atoms - pred_num_atoms), bins=50, color='tab:cyan')
    plt.axvline(x=np.mean(np.abs(true_num_atoms - pred_num_atoms)), linestyle='--', color='tab:red', label='Mean = %.3f' % np.mean(np.abs(true_num_atoms - pred_num_atoms)))
    plt.xlim(0, 10)
    plt.xlabel('$N_{true}$ - $N_{pred}$')
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.savefig('output/eval/' + mode + '/atoms.svg', format='svg')
    plt.close()

    x = np.linspace(0, 10, 100)
    plt.figure()
    plt.scatter(true_lc[:, 0], pred_lc[:,0], alpha=0.2, color='black')
    plt.plot(x, x, 'r--')
    plt.xlabel('$a$ True (Angstrom)')
    plt.ylabel('$a$ Pred (Angstrom)')
    plt.savefig('output/eval/' + mode + '/lattice_a_tp.svg', format='svg')
    plt.close()

    plt.figure()
    plt.scatter(true_lc[:, 1], pred_lc[:,1], alpha=0.2, color='black')
    plt.plot(x, x, 'r--')
    plt.xlabel('$b$ True (Angstrom)')
    plt.ylabel('$b$ Pred (Angstrom)')
    plt.savefig('output/eval/' + mode + '/lattice_b_tp.svg', format='svg')
    plt.close()

    plt.figure()
    plt.scatter(true_lc[:, 2], pred_lc[:,2], alpha=0.2, color='black')
    plt.plot(x, x, 'r--')
    plt.xlabel('$c$ True (Angstrom)')
    plt.ylabel('$c$ Pred (Angstrom)')
    plt.savefig('output/eval/' + mode + '/lattice_c_tp.svg', format='svg')
    plt.close()
