import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from cgcnn.cgcnn import CGCNN
from cgcnn.data import CifDataGenerator


data_dir = 'cgcnn/data/'
target = 'formation_energy_per_atom'
batch_size = 32
training_data = CifDataGenerator(data_dir, target, batch_size=batch_size, start_idx=0, end_idx=16384)
validation_data = CifDataGenerator(data_dir, target, batch_size=batch_size, start_idx=16384, end_idx=16384+2048)
print('training_data', len(training_data), 'validation_data', len(validation_data))


model = CGCNN(batch_size)
adam = Adam(learning_rate=1e-3)
checkpoint = ModelCheckpoint(filepath='saved_models/cgcnn_weights.best.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')
model.compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])

model.fit_generator(training_data, validation_data=validation_data, epochs=60, verbose=1, callbacks=[checkpoint])