import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from cgcnn.cgcnn import CGCNN
from cgcnn.data import CifDataGenerator
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='data_dir', default='data/cgcnn_cifs', type=str, help='Path to cif files')
    parser.add_argument('--batch_size', metavar='batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--ntrain', metavar='ntrain', default=16384, type=int, help='Number of training samples')
    parser.add_argument('--nval', metavar='nval', default=2048, type=int, help='Number of validation samples')
    parser.add_argument('--filepath', metavar='filepath', default='saved_models/cgcnn_weights.best.hdf5', type=str, help='Model save path')
    parser.add_argument('--target', metavar='tagrte', default='formation_energy_per_atom', type=str, help='Property to train')
    namespace = parser.parse_args()

    data_dir = namespace.data_dir
    batch_size = namespace.batch_size
    training_data = CifDataGenerator(data_dir, target=namespace.target, batch_size=batch_size, start_idx=0, end_idx=namespace.ntrain)
    validation_data = CifDataGenerator(data_dir, target=namespace.target, batch_size=batch_size, start_idx=namespace.ntrain, end_idx=namespace.ntrain+namespace.nval)
    print('training_data', len(training_data), 'validation_data', len(validation_data))


    model = CGCNN(batch_size)
    adam = Adam(learning_rate=1e-3)
    checkpoint = ModelCheckpoint(filepath=namespace.filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='min')
    model.compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])

    model.fit_generator(training_data, validation_data=validation_data, epochs=60, verbose=1, callbacks=[checkpoint])