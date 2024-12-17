import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Disable debugging and warning logs

from typing import Tuple

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import differential_evolution

from src.cnn_model import cnn_model


#%% Define ensemble class & associated utilities
class cnn_ensemble(cnn_model):

    # Initialize model parameters.
    def __init__(self, \
                 train_df: tf.Tensor, train_id: np.array, \
                 test_df: tf.Tensor, test_id: np.array, \
                 filter_size: int=24, kernel_size: int=5, \
                 dropout_rate: float=0.1, \
                 init_flag: bool=True) -> None:

        # Inherit parent utilities class.
        super().__init__(train_df=train_df, train_id=train_id, \
                         test_df=test_df, test_id=test_id)

        # Initialize MSTAR dataframe parameters.
        self.train_df = train_df                                                                        # Complex training data
        self.train_id = train_id                                                                        # Training data classification label

        self.test_df = test_df                                                                          # Complex test data
        self.test_id = test_id                                                                          # Test data classification label

        self.n_rg = self.train_df.shape[1]                                                              # Number of range pixels
        self.n_az = self.train_df.shape[2]                                                              # Number of azimuth pixels

        self.list_id = np.unique(self.train_id)                                                         # List of unique classifications
        self.n_id = self.list_id.size                                                                   # Number of unique classifications

        self.train_df_i = tf.convert_to_tensor(tf.math.real(self.train_df), dtype=tf.float32)           # Real component (I) of training data
        self.train_df_q = tf.convert_to_tensor(tf.math.imag(self.train_df), dtype=tf.float32)           # Imaginary component (Q) of training data

        self.test_df_i = tf.convert_to_tensor(tf.math.real(self.test_df), dtype=tf.float32)             # Real component (I) of test data
        self.test_df_q = tf.convert_to_tensor(tf.math.imag(self.test_df), dtype=tf.float32)             # Imaginary component (Q) of test data                       

        self.train_id_oh = tf.convert_to_tensor(self.to_one_hot(self.train_id), dtype=tf.float32)       # One-hot encoded training classification label
        self.test_id_oh = tf.convert_to_tensor(self.to_one_hot(self.test_id), dtype=tf.float32)         # One-hot encoded test classification label 

        self.train_id_sparse = tf.convert_to_tensor(np.reshape(np.nonzero(self.train_id_oh)[1][:], \
                                                               (self.train_id.size, 1)))                # Sparse training classification label
        self.test_id_sparse = tf.convert_to_tensor(np.reshape(np.nonzero(self.test_id_oh)[1][:], \
                                                              (self.test_id.size, 1)))                  # Sparse test classification label

        # Initialize model parameters.
        self.init_flag = init_flag                                                                      # Initialize new CNN model

        self.filter_size = filter_size                                                                  # Convolutional layer filter size
        self.kernel_size = kernel_size                                                                  # Convolutional layer kernel size
        self.dropout_rate = dropout_rate                                                                # Fraction of input units to drop

        # Initialize CNN ensemble.
        self.model_i = cnn_model(train_df=self.train_df, train_id=self.train_id, \
                                 test_df=self.test_df, test_id=self.test_id, \
                                 filter_size=self.filter_size, \
                                 kernel_size=self.kernel_size, \
                                 dropout_rate=self.dropout_rate, \
                                 iq_type='i', \
                                 init_flag=False)                                                       # Real component neural network (CNN) model
                        
        self.model_q = cnn_model(train_df=self.train_df, train_id=self.train_id, \
                                 test_df=self.test_df, test_id=self.test_id, \
                                 filter_size=self.filter_size, \
                                 kernel_size=self.kernel_size, \
                                 dropout_rate=self.dropout_rate, \
                                 iq_type='q', \
                                 init_flag=False)                                                       # Imaginary component neural network (CNN) model

        self.ensemble = [self.model_i, self.model_q]                                                    # I/Q CNN ensemble

        # Initialize average and weighted-average model weights.
        self.eq_wts = np.array((0.5, 0.5))                                                              # Average model weights

        if self.init_flag:

            self.opt_wts = self.get_opt_wts(self.test_df, self.test_id_oh)                              # Weighted average model weights
        
        else:

            self.opt_wts = np.array([0.48449787, 0.51550213])                                           # Weighted average model weights


    # Train underlying CNN models in ensemble.
    def train(self, \
              n_epoch: int=25, batch_size: int=64, patience: int=3, \
              iq_type: str='i', verbose: bool=True) -> None:
        
        # Define callbacks to optimize model training.
        es = EarlyStopping(monitor='val_loss', \
                           verbose=verbose, \
                           patience=patience, \
                           min_delta=0.001)                                 # Stop training at right time
        mc = ModelCheckpoint(str('cnn_' + iq_type.lower() + '.keras'), \
                             monitor='val_accuracy', \
                             verbose=verbose, \
                             save_best_only='True', \
                             mode='max')                                    # Save best model after each training epoch
        red_lr = ReduceLROnPlateau(monitor='val_loss', \
                                   factor=0.1,
                                   patience=int(patience/2),
                                   min_lr=0.0000001)                        # Reduce learning rate once learning stagnates
        
        # Perform model training.
        self.model.fit(x=self.train_df_i, y=self.train_id_oh, \
                    epochs=n_epoch, batch_size=batch_size, \
                    callbacks=[es, mc, red_lr], \
                    validation_data=(self.test_df_i, self.test_id_oh))

        self.model.fit(x=self.train_df_q, y=self.train_id_oh, \
                    epochs=n_epoch, batch_size=batch_size, \
                    callbacks=[es, mc, red_lr], \
                    validation_data=(self.test_df_q, self.test_id_oh))
        

    # Generate CNN ensemble predictions for multiple classification.
    def predict(self, x: tf.Tensor, w: np.array=None) -> float:
        
        if w is None:

            w = self.opt_wts

        x_i = tf.convert_to_tensor(tf.math.real(x), dtype=tf.float32)
        x_q = tf.convert_to_tensor(tf.math.imag(x), dtype=tf.float32)
        
        yhat = np.array((self.model_i.model.predict(x_i), self.model_q.model.predict(x_q)))     # Ensemble fitted values
        prob_sum = np.tensordot(yhat, w, axes=((0), (0)))                                       # Weighted sum across ensemble members

        return np.argmax(prob_sum, axis=1)
    

    # Evaluate CNN ensemble predictions.
    def evaluate(self, x: tf.Tensor, y: tf.Tensor, w: np.array=None, plot_flag: bool=False) -> float:

        if w is None:

            w = self.opt_wts
        
        yhat = self.predict(x, w)
        y = np.argmax(y, axis=1)

        if plot_flag:

            labels = np.array(['BMP2', 'BTR70', 'T72'])
            self.plot_confusion_matrix(y, yhat, labels)

        return accuracy_score(y, yhat)


    # Loss function for CNN ensemble weight optimization with differential evolution.
    def loss_fn(self, w: np.array, x: tf.Tensor, y: tf.Tensor) -> float:

        wn = self.norm(w)                   # Normalized ensemble weights
        score = self.evaluate(x, y, w=wn)   # Ensemble error rate

        return (1.0 - score)
    

    # Perform CNN ensemble weight optimization.
    def get_opt_wts(self, \
                    x: tf.Tensor, y: tf.Tensor, \
                    iter_max: int=10, epsilon: float=0.01) -> np.array:

        # Define optimization parameters.
        bound = [(0.0, 1.0), (0.0, 1.0)]
        search_arg = (x, y)

        # Perform global optimization of ensemble weights.
        opt_wts = differential_evolution(self.loss_fn, bound, \
                                         args=search_arg, maxiter=iter_max, tol=epsilon)
        
        return np.array(self.norm(opt_wts['x']))

