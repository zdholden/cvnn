import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Disable debugging and warning logs

from typing import Tuple

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import custom_object_scope


from cvnn.layers import complex_input
from cvnn.layers import ComplexConv2D, ComplexMaxPooling2D
from cvnn.layers import ComplexDropout, ComplexDense, ComplexFlatten

from cvnn.activations import cart_relu, cart_softmax
from cvnn.initializers import ComplexGlorotUniform

from sklearn.model_selection import GridSearchCV

from scikeras.wrappers import KerasClassifier

from utils.utils import utils
from utils.plot import plot
from utils.loss import ComplexAverageCrossEntropy, ComplexAverageSparseCrossEntropy


#%% Define model class & associated utilities
class cvnn_model(utils, plot):

    # Initialize model parameters.
    def __init__(self, \
                 train_df: tf.Tensor, train_id: np.array, \
                 test_df: tf.Tensor, test_id: np.array, \
                 filter_size: int=24, kernel_size: int=5, dropout_rate: float=0.1, \
                 init_flag: bool=True) -> None:

        # Inherit parent utilities class.
        super().__init__()

        # Initialize MSTAR dataframe parameters.
        self.train_df = train_df                                                                        # Complex training data
        self.train_id = train_id                                                                        # Training data classification label

        self.test_df = test_df                                                                          # Complex test data
        self.test_id = test_id                                                                          # Test data classification label

        self.n_rg = self.train_df.shape[1]                                                              # Number of range pixels
        self.n_az = self.train_df.shape[2]                                                              # Number of azimuth pixels

        self.list_id = np.unique(self.train_id)                                                         # List of unique classifications
        self.n_id = self.list_id.size                                                                   # Number of unique classifications

        self.train_id_oh = tf.convert_to_tensor(self.to_one_hot(self.train_id), dtype=tf.float32)       # One-hot encoded training classification label
        self.test_id_oh = tf.convert_to_tensor(self.to_one_hot(self.test_id), dtype=tf.float32)         # One-hot encoded test classification label 

        self.train_id_sparse = tf.convert_to_tensor(np.reshape(np.nonzero(self.train_id_oh)[1][:], \
                                                               (self.train_id.size, 1)))                # Sparse training classification label
        self.test_id_sparse = tf.convert_to_tensor(np.reshape(np.nonzero(self.test_id_oh)[1][:], \
                                                              (self.test_id.size, 1)))                  # Sparse test classification label

        # Initialize model parameters.
        self.init_flag = init_flag                                                                      # Initialize new CVNN model

        self.filter_size = filter_size                                                                  # Convolutional layer filter size
        self.kernel_size = kernel_size                                                                  # Convolutional layer kernel size
        self.dropout_rate = dropout_rate                                                                # Fraction of input units to drop

        # Initialize CVNN model.
        self.model = self.get_model(filter_size=self.filter_size, \
                                    kernel_size=self.kernel_size, \
                                    dropout_rate=self.dropout_rate,
                                    verbose=False)                                                      # Complex-valued neural network (CVNN) model                


    # Generate CVNN model.
    def get_model(self, \
                  filter_size: int=24, kernel_size: int=5, dropout_rate: float=0.1, \
                  verbose: bool=True) -> Model:

        # Create new CVNN model.
        with custom_object_scope({'complex_average_cross_entropy': ComplexAverageCrossEntropy, \
                                  'complex_average_sparse_cross_entropy': ComplexAverageSparseCrossEntropy}):
            
            if self.init_flag:

                # Input complex-valued image of shape [n_rg, n_az, 2].
                IN = complex_input(shape=(self.n_rg, self.n_az, 1))

                # Perform three convolutions of shape [n_rg, n_az, 2].
                C1 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(IN)
                C2 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(C1)
                C3 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(C2)

                # Downsample with max-pooling to shape [(n_rg/2), (n_az/2), 2] and drop random units.
                P1 = ComplexMaxPooling2D(pool_size=(2, 2), padding='same')(C3)
                D1 = ComplexDropout(rate=dropout_rate)(P1)
                
                # Perform three convolutions of shape [(n_rg/2), (n_az/2), 2].
                C4 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(D1)
                C5 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(C4)
                C6 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(C5)
                
                # Downsample with max-pooling to shape [(n_rg/4), (n_az/4), 2] and drop random units.
                P2 = ComplexMaxPooling2D(pool_size=(2, 2))(C6)
                D2 = ComplexDropout(rate=dropout_rate)(P2)

                # Perform three convolutions of shape [(n_rg/4), (n_az/4), 2].
                C7 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(D2)
                C8 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(C7)
                C9 = ComplexConv2D(filters=filter_size, \
                                activation=cart_relu, \
                                padding='same', \
                                kernel_size=(kernel_size, kernel_size), \
                                kernel_initializer=ComplexGlorotUniform)(C8)
                
                # Downsample with max-pooling to shape [(n_rg/8), (n_az/8), 2] and drop random units.
                P3 = ComplexMaxPooling2D(pool_size=(2, 2))(C9)
                D3 = ComplexDropout(rate=dropout_rate)(P3)

                # Flatten to shape [((n_rg/8) * (n_az/8) * 2), 1] and drop random units.
                F1 = ComplexFlatten()(D3)
                D4 = ComplexDropout(rate=dropout_rate)(F1)

                # Output one-hot encoded classification of shape [n_id, 1].
                DN = ComplexDense(self.n_id, activation=cart_softmax)(D4)
                OUT = Lambda(lambda x: tf.cast(tf.math.real(x), tf.float32))(DN)

                cvnn = Model(inputs=IN, outputs=OUT, name='ComplexConvNet')

                # Compile model for training.
                cvnn.compile(loss='categorical_crossentropy', \
                                optimizer=Adam(), \
                                metrics=['accuracy'])

            # Load existing CVNN model.
            else:

                cvnn = load_model('cvnn.keras', custom_objects={'ComplexInput': complex_input,
                                                                'ComplexConv2D': ComplexConv2D,
                                                                'ComplexFlatten': ComplexFlatten,
                                                                'ComplexDropout': ComplexDropout,
                                                                'ComplexDense': ComplexDense})

            # Print model summary, if desired.
            if verbose:

                cvnn.summary()

        return cvnn
    

    # Optimize CVNN model hyperparameters with cross-validated grid search.
    def gridsearch(self):

        with custom_object_scope({'complex_average_cross_entropy': ComplexAverageCrossEntropy, \
                                  'complex_average_sparse_cross_entropy': ComplexAverageSparseCrossEntropy}):

            # Generate model with KerasClassifier wrapper.
            model = KerasClassifier(build_fn=self.get_model())
            # scorer = make_scorer('complex_average_cross_entropy', greater_is_better=False)

            # Define the hyperparameters to search.

            search_grid = {'model__filter_size': [16, 32, 64], \
                           'model__kernel_size': [3, 5, 7], \
                           'model__dropout_rate': [0.1, 0.2, 0.3], \
                           'batch_size': [32, 64, 128]}

            # Perform cross-validated grid search and display results.
            grid = GridSearchCV(estimator=model, param_grid=search_grid, cv=3, n_jobs=-1)
            res = grid.fit(X=self.train_df, y=self.train_id_sparse)

            print('Cross-validated grid search hyperparameters: ' + res.best_params_ + ' (Score: ' + str(res.best_score_) + ')')
    

    # Train CVNN model on MSTAR dataset.
    def train(self, \
              n_epoch: int=25, batch_size: int=64, patience: int=3, \
              verbose: bool=True) -> None:
        
        # Define callbacks to optimize model training.
        es = EarlyStopping(monitor='val_loss', \
                           verbose=verbose, \
                           patience=patience, \
                           min_delta=0.001)                     # Stop training at right time
        mc = ModelCheckpoint('cvnn.keras', \
                             monitor='val_accuracy', \
                             verbose=verbose, \
                             save_best_only='True', \
                             mode='max')                        # Save best model after each training epoch
        red_lr = ReduceLROnPlateau(monitor='val_loss', \
                                   factor=0.1,
                                   patience=int(patience/2),
                                   min_lr=0.00001)              # Reduce learning rate once learning stagnates
        
        # Perform model training.
        self.model.fit(x=self.train_df, y=self.train_id_oh, \
                       epochs=n_epoch, batch_size=batch_size, \
                       callbacks=[es, mc, red_lr], \
                       validation_data=(self.test_df, self.test_id_oh))
        
        # Load best model from training.
        """
        self.model = load_model('cvnn.keras', custom_objects={'ComplexInput': complex_input,
                                                              'ComplexConv2D': ComplexConv2D,
                                                              'ComplexFlatten': ComplexFlatten,
                                                              'ComplexDropout': ComplexDropout,
                                                              'ComplexDense': ComplexDense})
        """


    # Generate CVNN model predictions for multiple classification.
    def predict(self, x: np.array):

        return np.argmax(self.model.predict(x), axis=1)
    

    # Evaluate CVNN model performance.
    def evaluate(self, \
                 x: tf.Tensor, y: tf.Tensor, \
                 batch_size: int=64, \
                 verbose: bool=True, plot_flag: bool=False) -> float:

        accuracy = self.model.evaluate(x, y, batch_size=batch_size, verbose=verbose)
        
        if plot_flag:
            
            yhat = self.predict(x)
            self.plot_confusion_matrix(y, yhat, np.array(['BMP2', 'BTR70', 'T72']))
        
        return accuracy[1]

