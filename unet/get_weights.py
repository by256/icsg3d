"""
ICSG3D/unet/get_weights.py
Function for calculating the class weights for the UNet loss
"""
import numpy as np
import os

def get_weights(path='', n_classes=95):
    if not path:
        return np.ones(n_classes)
    weights = np.zeros(n_classes)
    for f, fname in enumerate(os.listdir(os.path.join(path, 'species_matrices'))):
        if not fname.endswith('.npy') or '_rot_' in fname:
            continue
        S = np.load(path + '/species_matrices/' + fname)
        unique, counts = np.unique(S, return_counts=True)
        for i, un in enumerate(unique):
            weights[int(un)] += counts[i]
    normed_weights = np.sum(weights)/ weights
    normed_weights[normed_weights == np.inf] = 0
        
    return normed_weights

