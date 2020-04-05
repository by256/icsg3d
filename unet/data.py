"""
ICSG3D/unet/data.py
Classes for data generation on the UNet"""

import os

import numpy as np
from keras.utils import Sequence, to_categorical


class UnetDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_path, batch_size=2, dim=(32,32,32), n_channels=7,
                 n_classes=95, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_path=data_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, b = self.__data_generation(list_IDs_temp)

        return X, [y, b]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))
        B = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.create_lattice_meshgrid(ID, channels=self.n_channels)

            # Store class
            s = np.load(os.path.join(self.data_path, 'species_matrices', ID)).reshape((32,32,32,1))
            y[i,] = s
            B[i,] = np.where(s != 0, 1, 0)

        return X, to_categorical(y, num_classes=self.n_classes), B
    
    def create_lattice_meshgrid(self, ID, channels=4):
        M = np.load(os.path.join(self.data_path, 'density_matrices', ID)).reshape(1,32,32,32,1)
        if channels == 1:
            return M
        else:
            p = np.load(os.path.join(self.data_path, 'coordinate_grids', ID)).reshape(1,32,32,32,3)
            M_L_grid = np.concatenate((M, p), axis=-1)
            return M_L_grid
