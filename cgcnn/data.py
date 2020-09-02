import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
from keras.utils import Sequence
from pymatgen.core.structure import Structure


class AtomInitializer:
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class GaussianDistance:
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class CifDataGenerator(Sequence):
    def __init__(self, data_directory, target, batch_size=64, max_num_nbr=12, dmin=0, radius=8, step=0.2, pad_dim=50, shuffle=True, start_idx=None, end_idx=None):
        self.data_directory = data_directory
        self.target = target
        self.batch_size = batch_size
        self.max_num_nbr = max_num_nbr
        self.dmin = 0
        self.radius = 8
        self.step = 0.2
        self.pad_dim = pad_dim
        self.shuffle = shuffle
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        self.df = pd.read_csv(data_directory + 'structure-property-data.csv')
        self.df = self.df[self.df['nsites'] <= pad_dim]
        self.df = self.df.sample(frac=1, random_state=np.random.seed(9)).reset_index(drop=True)
        if self.end_idx:
            self.df = self.df.iloc[self.start_idx:self.end_idx, :]

        self.atom_init = AtomCustomJSONInitializer(self.data_directory + 'atom_init.json')
        self.gdf = GaussianDistance(dmin=self.dmin, dmax=self.radius, step=self.step)
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        indexes = np.arange(0, len(self.df))[idx*self.batch_size:(idx+1)*self.batch_size]
        
        atomic_features = []
        bond_features = []
        atom_neighbour_idxs = []
        masks = []
        targets = []

        for i in indexes:
            crystal = Structure.from_str(self.df['cif'].iloc[i], fmt='cif')

            atom_fea = np.vstack([self.atom_init.get_atom_fea(crystal[i].specie.number)
                                        for i in range(len(crystal))])
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    warnings.warn('{} not find enough neighbors to build graph. '
                                    'If it happens frequently, consider increase '
                                    'radius.'.format(self.df['mp_id'].iloc[i]))
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                        [0] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                    [self.radius + 1.] * (self.max_num_nbr -
                                                            len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)

            atom_fea_new, nbr_fea_new, nbr_fea_idx_new, mask_new = self.pad_features_with_n(atom_fea, nbr_fea, nbr_fea_idx, n=128)
            
            atomic_features.append(atom_fea_new)
            bond_features.append(nbr_fea_new)
            atom_neighbour_idxs.append(nbr_fea_idx_new)
            masks.append(mask_new)
            targets.append(self.df[self.target].iloc[i])

        atomic_features = np.array(atomic_features, dtype=np.float32)
        bond_features = np.array(bond_features, dtype=np.float32)
        atom_neighbour_idxs = np.array(atom_neighbour_idxs, dtype=int)
        masks = np.array(masks, dtype=int)
        targets = np.array(targets, dtype=np.float32)
        return [atomic_features, bond_features, atom_neighbour_idxs, masks], targets

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def pad_features_with_n(self, atom_features, bond_features, atom_neighbour_idxs, n=128):
        n_atoms = atom_features.shape[0]
        n_to_pad = self.pad_dim - n_atoms

        atom_padding = np.zeros(shape=(n_to_pad, atom_features.shape[1]))
        bond_padding = np.zeros(shape=(n_to_pad, bond_features.shape[1], bond_features.shape[2]))
        idx_padding = np.ones(shape=(n_to_pad, atom_neighbour_idxs.shape[1])) * (self.pad_dim - 1)

        atom_features = np.concatenate([atom_features, atom_padding], axis=0)
        bond_features = np.concatenate([bond_features, bond_padding], axis=0)
        atom_neighbour_idxs = np.concatenate([atom_neighbour_idxs, idx_padding], axis=0)

        mask_shape = list(atom_neighbour_idxs.shape) + [n]
        mask_ones = np.ones(shape=(n_atoms, atom_neighbour_idxs.shape[-1], n))
        mask_zeros = np.zeros(shape=(n_to_pad, atom_neighbour_idxs.shape[-1], n))
        mask = np.concatenate([mask_ones, mask_zeros], axis=0)
    
        return atom_features, bond_features, atom_neighbour_idxs, mask
    
    def less_than_n(self, x, n=50):
        filename = 'cifs/{}.cif'.format(x)
        crystal = Structure.from_file(filename)
        return len(crystal) < n


class PrecomputedDataGenerator(Sequence):
    def __init__(self, atomic_features, bond_features, atom_neighbour_idxs, masks, targets, batch_size):
        self.atomic_features = atomic_features
        self.bond_features = bond_features
        self.atom_neighbour_idxs = atom_neighbour_idxs
        self.masks = masks
        self.targets = targets
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.atomic_features) / self.batch_size))

    def __getitem__(self, idx):
        indexes = np.arange(0, len(self.atomic_features))[idx*self.batch_size:(idx+1)*self.batch_size]
        return [atomic_features[indexes], bond_features[indexes], atom_neighbour_idxs[indexes], masks[indexes]], targets[indexes]
