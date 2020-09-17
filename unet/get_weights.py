"""
## Compute class weights for the Unet loss function
## Each class is weighted inversely proportional to its frequency in the trainin set.
## Zerp class is always set to zero due to imbalance
--------------------------------------------------
## Author: Callum J. Court.
## Email: cc889@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Callum Court & Batuhan Yildirim 2020, ICSG3D
-------------------------------------------------
"""


import numpy as np
import os

def get_weights(path='', training_ids=[], n_classes=95):
    if not path:
        return np.ones(n_classes)
    weights = np.zeros(n_classes)
    for f, fname in enumerate(os.listdir(os.path.join(path, 'species_matrices'))):
        if not fname.endswith('.npy') or '_rot_' in fname or fname not in training_ids:
            continue
        S = np.load(path + '/species_matrices/' + fname)
        unique, counts = np.unique(S, return_counts=True)
        for i, un in enumerate(unique):
            weights[int(un)] += counts[i]
    normed_weights = np.sum(weights)/ weights
    normed_weights[normed_weights == np.inf] = 0
        
    return normed_weights

