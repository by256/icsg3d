import numpy as np
from .cgcnn import CGCNN
from .data import AtomInitializer, GaussianDistance, AtomCustomJSONInitializer
from pymatgen.core.structure import Structure
from keras.optimizers import Adam


def pad_features(atom_features, bond_features, atom_neighbour_idxs, pad_dim=50):
    n_atoms = atom_features.shape[0]
    n_to_pad = pad_dim - n_atoms

    atom_padding = np.zeros(shape=(n_to_pad, atom_features.shape[1]))
    bond_padding = np.zeros(shape=(n_to_pad, bond_features.shape[1], bond_features.shape[2]))
    idx_padding = np.ones(shape=(n_to_pad, atom_neighbour_idxs.shape[1])) * (pad_dim - 1)

    atom_features = np.concatenate([atom_features, atom_padding], axis=0)
    bond_features = np.concatenate([bond_features, bond_padding], axis=0)
    atom_neighbour_idxs = np.concatenate([atom_neighbour_idxs, idx_padding], axis=0)

    # compute mask
    mask_shape = list(atom_neighbour_idxs.shape) + [128]
    mask_ones = np.ones(shape=(n_atoms, atom_neighbour_idxs.shape[-1], 128))
    mask_zeros = np.zeros(shape=(n_to_pad, atom_neighbour_idxs.shape[-1], 128))
    mask = np.concatenate([mask_ones, mask_zeros], axis=0)

    
    return atom_features, bond_features, atom_neighbour_idxs, mask


def evaluate_cgcnn_from_cif(model, cif, weights, batch_size, atom_init='./cgcnn/atom_init_cjc.json', weights_dir='saved_models/cgcnn'):

    atom_init = AtomCustomJSONInitializer(atom_init)
    max_num_nbr = 12
    dmin = 0
    radius = 8
    step = 0.2
    gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
    pad_dim = 50


    if cif.endswith('.cif'):
        # load crystal from .cif file
        crystal = Structure.from_file(cif)
    else:
        # load crystal from cif string
        with open('temp.cif', 'w') as f:
            f.write(df.iloc[i]['cif'])
        crystal = Structure.from_file(base_path + 'temp.cif')
    if len(crystal) > 50:
        pass
    atom_fea = np.vstack([atom_init.get_atom_fea(crystal[i].specie.number)
                                for i in range(len(crystal))])
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                            [radius + 1.] * (max_num_nbr -
                                                    len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],
                                    nbr[:max_num_nbr])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)

    if atom_fea.shape[0] != pad_dim:
        atom_fea, nbr_fea, nbr_fea_idx, mask = pad_features(atom_fea, nbr_fea, nbr_fea_idx, pad_dim=pad_dim)
    else:
        mask = np.ones(shape=(pad_dim, nbr_fea_idx.shape[-1], 128))

    atom_fea = np.expand_dims(atom_fea, axis=0)
    nbr_fea = np.expand_dims(nbr_fea, axis=0)
    nbr_fea_idx = np.expand_dims(nbr_fea_idx, axis=0)
    mask = np.expand_dims(mask, axis=0)
    preds = []
    for prop in weights:
        weights_path = weights_dir + '/cgcnn_weights.' + prop +'.best.hdf5'
        model.load_weights(weights_path)
        adam = Adam(lr=0.001)
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_absolute_error'])


        data = {'atom_input': atom_fea, 
                'bond_input': nbr_fea, 
                'atom_n_input': nbr_fea_idx, 
                'masks_input': mask}
        pred = model.predict(data, batch_size=batch_size)
        preds.append(pred)
    return preds