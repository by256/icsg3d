"""
ICSG3D/vae/data.py
Data generation classes for the VAE"""

import os
import re

import numpy as np
import pandas as pd
from keras.utils import Sequence, to_categorical


class VAEDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_path, batch_size=2, dim=(32,32,32), n_channels=4,
                 n_classes=95, shuffle=False, property_csv='property.csv', n_bins=10, target='formation_energy_per_atom', return_S=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_path=data_path
        self.property_df = pd.read_csv(property_csv)
        self.n_bins = n_bins
        self.n_atoms = self.property_df['nsites'].max()+1
        labels = np.arange(n_bins)
        self.property_df['bin'] = pd.qcut(self.property_df[target], self.n_bins, labels).astype(int)
        self.on_epoch_end()
        self.return_S = return_S

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        self.indexes_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in self.indexes_temp]

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        M = np.empty((self.batch_size, *self.dim, self.n_channels))
        cond = np.empty((self.batch_size, self.n_bins))
        if self.return_S:
            S = np.empty((self.batch_size, *self.dim, 1))
            S_b = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(self.list_IDs_temp):
            # Store sample:
            M[i,] = self.create_lattice_meshgrid(ID, channels=self.n_channels)
            cond[i,] = self.property_to_categorical(ID)
            if self.return_S:
                S[i,] = np.load(os.path.join(self.data_path, 'species_matrices', ID)).reshape(1,32,32,32,1)
                S_b[i,] = np.where(S[i,] != 0, 1, 0)
        if self.return_S:
            return M, [cond, to_categorical(S, num_classes=self.n_classes), S_b]
        else:
            return M, cond
    
    def property_to_categorical(self, ID):
        cif_id = re.split('_|\.', ID)[0]
        bin_num = self.property_df[self.property_df['task_id'] == cif_id]['bin'].values
        return to_categorical(bin_num, num_classes=self.n_bins)

    def create_lattice_meshgrid(self, ID, channels=4):
        M = np.load(os.path.join(self.data_path, 'density_matrices', ID)).reshape(1,32,32,32,1)
        if channels == 1:
            return M
        else:
            p = np.load(os.path.join(self.data_path, 'coordinate_grids', ID)).reshape(1,32,32,32,3)
            M_L_grid = np.concatenate((M, p), axis=-1)
            return M_L_grid
