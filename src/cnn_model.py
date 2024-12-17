import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Disable debugging and warning logs

from typing import Tuple

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer

from utils.utils import utils
from utils.plot import plot


#%% Define model class & associated utilities
class cnn_model(utils, plot):

    # Initialize model parameters.
    def __init__(self, \
                 train_df: tf.Tensor, train_id: np.array, \
                 test_df: tf.Tensor, test_id: np.array, \
                 filter_size: int=24, kernel_size: int=5, dropout_rate: float=0.5, \
                 iq_type: str='i', init_flag: bool=True) -> None:

        # Inherit parent utilities class.
        super().__init__()

        # Initialize MSTAR dataframe parameters.
        self.train_df = train_df                                                                        # Complex training data
        self.train_id = train_id                                                                        # Training data classification label

        self.test_df = test_df                                                                          # Complex test data
        self.test_id = test_id                                                                          # Test data classification label

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

        self.n_rg = self.train_df.shape[1]                                                              # Number of range pixels
        self.n_az = self.train_df.shape[2]                                                              # Number of azimuth pixels

        self.list_id = np.unique(self.train_id)                                                         # List of unique classifications
        self.n_id = self.list_id.size                                                                   # Number of unique classifications


        # Initialize model parameters.
        self.init_flag = init_flag                                                                      # Initialize new CNN model

        self.filter_size = filter_size                                                                  # Convolutional layer filter size
        self.kernel_size = kernel_size                                                                  # Convolutional layer kernel size
        self.dropout_rate = dropout_rate                                                                # Fraction of input units to drop

        self.iq_type = iq_type                                                                          # Phase component type (I/Q)

        # Initialize CNN model.
        self.model = self.get_model(filter_size=self.filter_size, \
                                    kernel_size=self.kernel_size, \
                                    dropout_rate=self.dropout_rate,
                                    verbose=False)                                                      # Real-valued neural network (CNN) model                


    # Generate CNN model.
    def get_model(self, \
                  filter_size: int=24, kernel_size: int=5, dropout_rate: float=0.1, \
                  verbose: bool=True) -> Model:

        # Create new CNN model.  
        if self.init_flag:

            # Input complex-valued image of shape [n_rg, n_az, 2].
            IN = Input(shape=(self.n_rg, self.n_az, 1))

            # Perform three convolutions of shape [n_rg, n_az, 2].
            C1 = Conv2D(filters=filter_size, \
                        activation='relu', \
                        padding='same', \
                        kernel_size=(kernel_size, kernel_size), \
                        kernel_initializer='he_normal')(IN)
            C2 = Conv2D(filters=filter_size, \
                        activation='relu', \
                        padding='same', \
                        kernel_size=(kernel_size, kernel_size), \
                        kernel_initializer='he_normal')(C1)
            C3 = Conv2D(filters=filter_size, \
                        activation='relu', \
                        padding='same', \
                        kernel_size=(kernel_size, kernel_size), \
                        kernel_initializer='he_normal')(C2)

            # Downsample with max-pooling to shape [(n_rg/2), (n_az/2), 2] and drop random units.
            P1 = MaxPooling2D(pool_size=(2, 2), padding='same')(C3)
            D1 = Dropout(rate=dropout_rate)(P1)
            
            # Perform three convolutions of shape [(n_rg/2), (n_az/2), 2].
            C4 = Conv2D(filters=filter_size, \
                        activation='relu', \
                        padding='same', \
                        kernel_size=(kernel_size, kernel_size), \
                        kernel_initializer='he_normal')(D1)
            C5 = Conv2D(filters=filter_size, \
                        activation='relu', \
                        padding='same', \
                        kernel_size=(kernel_size, kernel_size), \
                        kernel_initializer='he_normal')(C4)
            C6 = Conv2D(filters=filter_size, \
                        activation='relu', \
                        padding='same', \
                        kernel_size=(kernel_size, kernel_size), \
                        kernel_initializer='he_normal')(C5)
            
            # Downsample with max-pooling to shape [(n_rg/4), (n_az/4), 2] and drop random units.
            P2 = MaxPooling2D(pool_size=(2, 2))(C6)
            D2 = Dropout(rate=dropout_rate)(P2)

            # Perform three convolutions of shape [(n_rg/4), (n_az/4), 2].
            C7 = Conv2D(filters=filter_size, \
                            activation='relu', \
                            padding='same', \
                            kernel_size=(kernel_size, kernel_size), \
                            kernel_initializer='he_normal')(D2)
            C8 = Conv2D(filters=filter_size, \
                            activation='relu', \
                            padding='same', \
                            kernel_size=(kernel_size, kernel_size), \
                            kernel_initializer='he_normal')(C7)
            C9 = Conv2D(filters=filter_size, \
                            activation='relu', \
                            padding='same', \
                            kernel_size=(kernel_size, kernel_size), \
                            kernel_initializer='he_normal')(C8)
            
            # Downsample with max-pooling to shape [(n_rg/8), (n_az/8), 2] and drop random units.
            P3 = MaxPooling2D(pool_size=(2, 2))(C9)
            D3 = Dropout(rate=dropout_rate)(P3)

            # Flatten to shape [((n_rg/8) * (n_az/8) * 2), 1] and drop random units.
            F1 = Flatten()(D3)
            D4 = Dropout(rate=dropout_rate)(F1)

            # Output one-hot encoded classification of shape [n_id, 1].
            OUT = Dense(self.n_id, activation='softmax')(D4)

            cnn = Model(inputs=IN, outputs=OUT, name=str('ConvNet' + self.iq_type.upper()))

            # Compile model for training.
            cnn.compile(loss='categorical_crossentropy', \
                            optimizer=Adam(learning_rate=0.0001), \
                            metrics=['accuracy'])

        # Load existing CVNN model.
        else:

            cnn = load_model(str('cnn_' + self.iq_type.lower() + '.keras'))

        # Print model summary, if desired.
        if verbose:

            cnn.summary()

        return cnn
    

    # Train CNN model on MSTAR dataset.
    def train(self, \
              n_epoch: int=25, batch_size: int=64, patience: int=3, \
              verbose: bool=True) -> None:
        
        # Define callbacks to optimize model training.
        es = EarlyStopping(monitor='val_loss', \
                           verbose=verbose, \
                           patience=patience, \
                           min_delta=0.001)                                 # Stop training at right time
        mc = ModelCheckpoint(str('cnn_' + self.iq_type.lower() + '.keras'), \
                             monitor='val_accuracy', \
                             verbose=verbose, \
                             save_best_only='True', \
                             mode='max')                                    # Save best model after each training epoch
        red_lr = ReduceLROnPlateau(monitor='val_loss', \
                                   factor=0.1,
                                   patience=int(patience/2),
                                   min_lr=0.0000001)                        # Reduce learning rate once learning stagnates
        
        # Perform model training.
        if (self.iq_type.lower() == 'i'):

            self.model.fit(x=self.train_df_i, y=self.train_id_oh, \
                        epochs=n_epoch, batch_size=batch_size, \
                        callbacks=[es, mc, red_lr], \
                        validation_data=(self.test_df_i, self.test_id_oh))
        
        elif (self.iq_type.lower() == 'q'):

            self.model.fit(x=self.train_df_q, y=self.train_id_oh, \
                        epochs=n_epoch, batch_size=batch_size, \
                        callbacks=[es, mc, red_lr], \
                        validation_data=(self.test_df_q, self.test_id_oh))
        
        else:

            raise ValueError('Invalid phase component type for real-valued model.')

