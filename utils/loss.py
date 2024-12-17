import tensorflow as tf
import keras

from tensorflow.keras import backend
from tensorflow.keras.losses import Loss, categorical_crossentropy, SparseCategoricalCrossentropy


@tf.keras.saving.register_keras_serializable(package='loss')
class ComplexAverageCrossEntropy(Loss):

    def call(self, y_true, y_pred):

        if y_true.dtype.is_complex:

            real_loss = categorical_crossentropy(tf.math.real(y_true), tf.math.real(y_pred)) \
                         + categorical_crossentropy(tf.math.imag(y_true), tf.math.real(y_pred))
            
            if y_pred.dtype.is_complex:

                imag_loss = categorical_crossentropy(tf.math.real(y_true), tf.math.imag(y_pred)) \
                         + categorical_crossentropy(tf.math.imag(y_true), tf.math.imag(y_pred))
            
            else:

                imag_loss = real_loss

        else:

            real_loss = categorical_crossentropy(y_true, tf.math.real(y_pred))
            
            if y_pred.dtype.is_complex:
                
                imag_loss = categorical_crossentropy(y_true, tf.math.imag(y_pred))
            
            else:
                
                imag_loss = real_loss
        
        return ((real_loss + imag_loss) / 2.0)
    

@tf.keras.saving.register_keras_serializable(package='loss')
class ComplexAverageSparseCrossEntropy(Loss):

    def call(self, y_true, y_pred):

        sparse_categorical_crossentropy = SparseCategoricalCrossentropy(from_logits=False)

        if y_true.dtype.is_complex:

            real_loss = sparse_categorical_crossentropy(tf.math.real(y_true), tf.math.real(y_pred)) \
                         + sparse_categorical_crossentropy(tf.math.imag(y_true), tf.math.real(y_pred))
            
            if y_pred.dtype.is_complex:

                imag_loss = sparse_categorical_crossentropy(tf.math.real(y_true), tf.math.imag(y_pred)) \
                         + sparse_categorical_crossentropy(tf.math.imag(y_true), tf.math.imag(y_pred))
            
            else:

                imag_loss = real_loss

        else:

            real_loss = sparse_categorical_crossentropy(y_true, tf.math.real(y_pred))
            
            if y_pred.dtype.is_complex:
                
                imag_loss = sparse_categorical_crossentropy(y_true, tf.math.imag(y_pred))
            
            else:
                
                imag_loss = real_loss
        
        return ((real_loss + imag_loss) / 2.0)
    
