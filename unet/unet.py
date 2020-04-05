
"""
ICSG3D/unet/unet.py
UNet based semantic segmentation network
"""
import os
import time
import warnings

import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.layers import *
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model, load_model, model_from_json
from keras.optimizers import Adam
from keras.regularizers import l2

from keras_contrib.layers.normalization.instancenormalization import \
    InstanceNormalization
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from unet.get_weights import get_weights

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

class TrainingPlot(Callback):

    def __init__(self, val_gen, sdir):
        super(TrainingPlot, self).__init__()
        self.val_gen = val_gen
        self.min_val_loss = np.inf
        self.sdir = sdir

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.precision = []
        self.val_losses = []
        self.val_precision = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        if self.val_losses[-1] < self.min_val_loss:
            self.min_val_loss = self.val_losses[-1]
            self.plot_segmentations(epoch, name=os.path.join(self.sdir, 'segmentations_best.png'))
            self.plot_binary(epoch, name=os.path.join(self.sdir, 'segmentations_binary_best.png'))
        self.plot_losses(epoch, logs)
    
    def plot_losses(self, epoch, logs={}):

        # Before plotting ensure at least 2 epochs have passed
        N = np.arange(0, len(self.losses))

        # You can chose the style of your preference
        # print(plt.style.available) to see the available options
        #plt.style.use("seaborn")

        # Plot train loss, train acc, val loss and val acc against epochs passed
        plt.figure()
        plt.plot(N, self.losses, label = "train_loss")
        plt.plot(N, self.val_losses, label = "val_loss")
        plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        # Make sure there exists a folder called output in the current directory
        # or replace 'output' with whatever direcory you want to put in the plots
        plt.savefig(os.path.join(self.sdir, 'unet_loss.png'))
        plt.close()
    
    def plot_segmentations(self, epoch, logs={}, name=None):
        fig1, axes1 = plt.subplots(10, 2)
        ax = 0
        for M, t in self.val_gen:
            S = t[0]
            recon_multi, recon_b = self.model.predict(M)
            recon_multi = np.argmax(recon_multi, axis=-1).reshape((len(M),32,32,32,1))
            S = np.argmax(S, axis=-1).reshape((len(M),32,32,32,1))
            axes1[ax][0].imshow(S[0, :,:,16,0], vmin=np.min(S[0,:,:,16,0]), vmax=np.max(S[0,:,:,16,0]))
            axes1[ax][1].imshow(recon_multi[0,:,:,16,0], vmin=np.min(recon_multi[0,:,:,16,0]), vmax=np.max(recon_multi[0,:,:,16,0]))
            axes1[ax][0].set_xticks([])
            axes1[ax][0].set_yticks([])
            axes1[ax][1].set_xticks([])
            axes1[ax][1].set_yticks([])
            ax += 1

            if ax == 10:
                break
        if name is None:
            name = 'output/unet/segmentations_epoch_%d.png' % epoch

        plt.savefig(name)
        plt.close()
        return
    
    def plot_binary(self, epoch, logs={}, name=None):
        fig1, axes1 = plt.subplots(10, 2)
        ax = 0
        for M, t in self.val_gen:
            B = t[1]
            recon_multi, recon_b = self.model.predict(M)
            B = B.reshape((len(M), 32,32,32,1))
            axes1[ax][0].imshow(B[0,:,:,16,0], vmin=np.min(B[0,:,:,16,0]), vmax=np.max(B[0,:,:,16,0]))
            axes1[ax][1].imshow(recon_b[0,:,:,16,0], vmin=np.min(recon_b[0,:,:,16,0]), vmax=np.max(recon_b[0,:,:,16,0]))
            axes1[ax][0].set_xticks([])
            axes1[ax][0].set_yticks([])
            axes1[ax][1].set_xticks([])
            axes1[ax][1].set_yticks([])
            ax += 1

            if ax == 10:
                break
        if name is None:
            name = 'output/unet/segmentations_epoch_%d.png' % epoch

        plt.savefig(name)
        plt.close()
        return

def r_m(y_true, y_pred):
    """
    Computes the average per-column recall metric
    for a multi-class classification problem
    """ 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
    recall = true_positives / (possible_positives + K.epsilon())    
    return recall

def wr_m(y_true, y_pred):
    """
    Weighted Recall, removes the zero class from calculations
    """
    weights = np.ones(95)
    weights[0] = 0.0
    true_positives = K.sum(K.round(K.clip(weights * y_true * y_pred, 0, 1)))  
    possible_positives = K.sum(K.round(K.clip(weights * y_true, 0, 1)))   
    recall = true_positives / (possible_positives + K.epsilon())    
    return recall

def p_m(y_true, y_pred):
    """ Precision metric """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """ F1-score metric"""
    precision = p_m(y_true, y_pred)
    recall = r_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, axis=-1)  # Total per voxel CCE (batch, 32,32,32)
        return K.mean(loss, axis=(1,2,3))
    
    return loss

class AtomUnet():
    def __init__(self, num_classes=95, class_weights=None, weights=None, input_shape=(32,32,32,4), lr=1e-6):
        self.class_weights = class_weights
        self.input_shape = input_shape
        self.optimizer = Adam(lr)
        self.num_classes = num_classes
        self.model = self.unet_3d_multiclass(num_classes)

        self.metrics = {'soft': [f1_m, wr_m]}
        self.metric_names = ['Loss', 'lsoft', 'lsig', 'f1', 'wr']
        
        self.model.compile(loss={'soft': weighted_categorical_crossentropy(num_classes), 'sig': 'binary_crossentropy'},
                           optimizer=self.optimizer,
                          metrics=self.metrics)
                          
        if weights and os.path.exists(weights):
            self.model.load_weights(weights)
            print("loaded weights")
            self.filepath = weights
        elif weights and not os.path.exists(weights):
            self.filepath = weights
        else:
            self.filepath = './saved_models/unet_%d_channel_weights.best.hdf5' % input_shape[-1]
    
    def unet_3d_multiclass(self, classes):
        inp = Input(shape=self.input_shape, name='unet_input')
        # Down
        # Layer 1
        c1 = Conv3D(filters=32, kernel_size=(3,3,3), padding='same')(inp)
        c1 = ReLU()(c1)
        c1 = BatchNormalization()(c1)
        c2 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same')(c1)
        c2 = ReLU()(c2)
        c2 = BatchNormalization()(c2)
        pool1 = MaxPool3D(strides=(2,2,2))(c2)
        
        # Layer 2
        c3 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same')(pool1)
        c3 = ReLU()(c3)
        c3 = BatchNormalization()(c3)
        c4 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same')(c3)
        c4 = ReLU()(c4)
        c4 = BatchNormalization()(c4)
        pool2 = MaxPool3D(strides=(2,2,2))(c4)

        # # Layer 3
        c5 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same')(pool2)
        c5 = ReLU()(c5)
        c5 = BatchNormalization()(c5)
        c6 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same')(c5)
        c6 = ReLU()(c6)
        c6 = BatchNormalization()(c6)
        pool3 = MaxPool3D(strides=(2,2,2))(c6)

        #         Layer 4
        # c7 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same')(pool3)
        # c7 = ReLU()(c7)
        # c7 = BatchNormalization()(c7)
        # c8 = Conv3D(filters=512, kernel_size=(3,3,3), padding='same')(c7)
        # c8 = ReLU()(c8)
        # c8 = BatchNormalization()(c8)
        # pool4 = MaxPool3D(strides=(2,2,2))(c8)

        # Bottom
        c9 = Conv3D(filters=512, kernel_size=(3,3,3), padding='same')(pool3)
        c9 = ReLU()(c9)
        c9 = BatchNormalization()(c9)
        c10 = Conv3D(filters=512, kernel_size=(3,3,3), padding='same')(c9)
        c10 = ReLU()(c10)
        c10 = BatchNormalization()(c10)
        up1 = UpSampling3D(size=(2,2,2))(c10)
        
        #         Up layer 4
        # concat1 = concatenate([c8, up1], axis=-1)
        # c11 = Conv3D(filters=1024, kernel_size=(3,3,3), padding='same')(concat1)
        # c11 = ReLU()(c11)
        # c11 = BatchNormalization()(c11)
        # c12 = Conv3D(filters=512, kernel_size=(3,3,3), padding='same')(c11)
        # c12 = ReLU()(c12)
        # c12 = BatchNormalization()(c12)
        # up2 = UpSampling3D(size=(2,2,2))(c12)

        #         Up layer 3
        concat2 = concatenate([c6, up1], axis=-1)
        c13 = Conv3D(filters=512, kernel_size=(3,3,3), padding='same')(concat2)
        c13 = ReLU()(c13)
        c13 = BatchNormalization()(c13)
        c14 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same')(c13)
        c14 = ReLU()(c14)
        c14 = BatchNormalization()(c14)
        up3 = UpSampling3D(size=(2,2,2))(c14)

        # Up layer 2
        concat3 = concatenate([c4, up3], axis=-1)
        c15 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same')(concat3)
        c15 = ReLU()(c15)
        c15 = BatchNormalization()(c15)
        c16 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same')(c15)
        c16 = ReLU()(c16)
        c16 = BatchNormalization()(c16)
        up4 = UpSampling3D(size=(2,2,2))(c16)

        # Up layer 1
        concat4 = concatenate([c2, up4], axis=-1)
        c17 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same')(concat4)
        c17 = ReLU()(c17)
        c17 = BatchNormalization()(c17)
        c18 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same')(c17)
        c18 = ReLU()(c18)
        c18 = BatchNormalization()(c18)
        multi_class_output = Conv3D(filters=int(classes), kernel_size=(1,1,1), padding='same', activation='softmax', name='soft')(c18)
        binary_output = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='sigmoid', name='sig')(c18)
        model = Model(inp, [multi_class_output, binary_output], name='unet')
        #print(model.summary())
        return model
    
    def train_generator(self, train_gen, val_gen, epochs=100, output_dir='output/unet/'):
        print("Training...")
        checkpoint1 = ModelCheckpoint(self.filepath, monitor='val_soft_loss', verbose=1, save_best_only=True, mode='min')
        plotter = TrainingPlot(val_gen, output_dir)
        callbacks_list = [checkpoint1, plotter]
        self.model.fit_generator(generator=train_gen,
                    validation_data=val_gen,
                    use_multiprocessing=False,
                    workers=4, epochs=epochs, callbacks=callbacks_list)
        self.model.load_weights(self.filepath)
        self.model.save(os.path.splitext(self.filepath)[0] + '.h5')
        print("Model saved")
        return
    
    def predict_generator(self, test_gen):
        result = self.model.predict_generator(test_gen)
        return result
    
    def save_(self, weights, model='saved_models/unet.h5'):
        self.model.load_weights(weights)
        self.model.save(model)
        return

# Weights for perceptual model
class_weights = get_weights()
custom_objects={'loss':  weighted_categorical_crossentropy(class_weights), 'f1_m': f1_m, 'wr_m': wr_m}
