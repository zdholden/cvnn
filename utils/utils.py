import os

import warnings

warnings.filterwarnings('error', category=RuntimeWarning)

from typing import Tuple
from glob import glob
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

import matplotlib.pyplot as plt

import struct


# Define built-in utilities class.
class utils():

    # Extract header and complex floating point data from MSTAR file.
    def unpack_mstar_file(self, fname: str) -> Tuple[dict, np.array]:
    
        # Initialize output dictionary for header.
        header = {}

        # Parse MSTAR file.
        with open(fname, 'rb') as f:

            # Unpack Phoenix-formatted ASCII header.
            for bline in f:

                sline = str(bline.decode('ascii'))  # Decode ASCII bytes to string

                if sline.startswith('[EndofPhoenixHeader]'):

                    break

                else:

                    try:
                        
                        # Define key-value pair for output dictionary.
                        key, val = sline.split('= ')
                        
                        key = key.strip()
                        val = val.strip()

                        # Convert to numerical data type, if applicable.
                        if val.isdigit():

                            val = int(val)

                        elif '.' in val:

                            try:

                                val = float(val)

                            except ValueError:

                                pass

                        header[key] = val

                    except(ValueError):

                        pass
            
            # Unpack complex floating point data formatted in magnitude and phase blocks.
            mag = np.zeros((header['NumberOfRows'], header['NumberOfColumns']), dtype=np.float32)
            phs = np.zeros((header['NumberOfRows'], header['NumberOfColumns']), dtype=np.float32)

            data = np.zeros((header['NumberOfRows'], header['NumberOfColumns']), dtype=np.complex64)

            for i in range(2):

                for m in range(data.shape[0]):
                    
                    for n in range(data.shape[1]):
                        
                        if (i == 0):
                            
                            mag[m, n] = struct.unpack('!f', f.read(4))[0]   # Single-precision (32-bit/4-byte) float with big-endian byte order

                        else:

                            phs[m, n] = struct.unpack('!f', f.read(4))[0]   # Single-precision (32-bit/4-byte) float with big-endian byte order
            
            data.real = (mag * np.cos(phs))
            data.imag = (mag * np.sin(phs))

        return (header, data)
    

    # Generate list of MSTAR files.
    def get_mstar_file_list(self, subset: str=None) -> np.array:

        if (subset.lower() == 'train'):

            return np.array(glob(os.path.join(os.getcwd(), '*/TARGETS/TRAIN/*/*/*/*.0*')))  # Training subset of dataset
        
        elif (subset.lower() == 'test'):

            return np.array(glob(os.path.join(os.getcwd(), '*/TARGETS/TEST/*/*/*/*.0*')))   # Test subset of dataset
        
        else:

            return np.array(glob(os.path.join(os.getcwd(), '*/TARGETS/*/*/*/*/*.0*')))      # Complete dataset


    # Smooth NaN values in complex images.
    def smooth_nan(self, x: np.array) -> np.array:

        # Generate mask of NaN values in image.
        mask = np.nonzero(np.isnan(np.squeeze(x)))
        
        # Smooth NaN pixels as average of non-NaN neighboring pixels, if possible.
        m = [-1, 0, 1]
        xm = []

        for i in range(mask[0].size):
            
            for m0 in range(3):

                for m1 in range(3):

                    try:

                        xm.append(x[mask[0][i]+m[m0]][mask[1][i]+m[m1]])
                
                    except(IndexError):

                        pass

            x[mask[0][i]][mask[1][i]] = np.mean(~np.isnan(xm))
            
        return x
        
    
    # One-hot encode classification labels.
    def to_one_hot(self, y: np.array) -> np.array:

        # Determine number of classification labels.
        n_id = np.unique(y).size

        # Convert standard labels to integer encoded labels.
        le = LabelEncoder()
        
        ydot = le.fit_transform(y=y)
        ydot.shape = (ydot.shape[0], 1)

        # Convert integer encoded labels to one-hot encoded labels.
        ydot = to_categorical(ydot, num_classes=n_id)

        return ydot
    

    # Convert complex pixel values to 0-255 image grayscale.
    def to_imscale(self, \
                   x_train: np.array, x_test: np.array, \
                   t_min: float=0.0, t_max: float=255.0) -> Tuple[np.array, np.array]:
        
        # Smooth NaN values in real/imaginary components of training data.
        pbar_train = tqdm(range(x_train.shape[0]))
        pbar_train.set_description('Smoothing NaN pixels in training data')

        for i in pbar_train:

            x_train[i].real = self.smooth_nan(x_train[i].real)
            x_train[i].imag = self.smooth_nan(x_train[i].imag)
        
        # Smooth NaN values in real/imaginary components of test data.
        pbar_test = tqdm(range(x_test.shape[0]))
        pbar_test.set_description('Smoothing NaN pixels in test data')
        
        for j in pbar_test:

            x_test[j].real = self.smooth_nan(x_test[j].real)
            x_test[j].imag = self.smooth_nan(x_test[j].imag)

        # Determine global minimum and maximum of real/imaginary components.
        re_min = np.min((np.min(x_train.real), np.min(x_test.real)))
        re_max = np.max((np.max(x_train.real), np.max(x_test.real)))

        im_min = np.min((np.min(x_train.imag), np.min(x_test.imag)))
        im_max = np.max((np.max(x_train.imag), np.max(x_test.imag)))

        x_min = np.min((re_min, im_min))
        x_max = np.max((re_max, im_max))

        # Independently scale real/imaginary components.
        try:

            # x_train.real = (((x_train.real - re_min) / (re_max - re_min)) * (t_max - t_min)) + t_min
            x_train.real = (((x_train.real - x_min) / (x_max - x_min)) * (t_max - t_min)) + t_min

        except (RuntimeWarning):

            x_train.real = 255.0
        
        try:

            # x_test.real = (((x_test.real - re_min) / (re_max - re_min)) * (t_max - t_min)) + t_min
            x_test.real = (((x_test.real - x_min) / (x_max - x_min)) * (t_max - t_min)) + t_min
        
        except (RuntimeWarning):

            x_test.real = 255.0

        try:

            # x_train.imag = (((x_train.imag - im_min) / (im_max - im_min)) * (t_max - t_min)) + t_min
            x_train.imag = (((x_train.imag - x_min) / (x_max - x_min)) * (t_max - t_min)) + t_min
        
        except (RuntimeWarning):

            x_train.imag = 255.0
        
        try:

            # x_test.imag = (((x_test.imag - im_min) / (im_max - im_min)) * (t_max - t_min)) + t_min
            x_test.imag = (((x_test.imag - x_min) / (x_max - x_min)) * (t_max - t_min)) + t_min

        except (RuntimeWarning):

            x_test.imag = 255.0

        return (x_train, x_test)
    

    # Convert NumPy array to Tensorflow tensor.
    def to_tensor(self, x_train: np.array, x_test: np.array) -> Tuple[tf.Tensor, tf.Tensor]:

        return (tf.constant(x_train, dtype=tf.complex64), tf.constant(x_test, dtype=tf.complex64))
    

    # Normalize vector x to unit length.
    def norm(self, x: np.array):

        # Calculate L1 vector norm.
        xn = np.linalg.norm(x, 1)

        if (xn == 0.0):

            return x
        
        else:

            return (x / xn)
    
