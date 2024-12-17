import os
import itertools

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# Define built-in plot class.
class plot():

    # Plot heatmap of MSTAR complex image.
    def plot_heatmap(self, \
                     sicd: np.array, title: str=None, xlabel: str=None, ylabel: str=None, \
                     cmap: str='gray', origin: str='lower', \
                     fpath: str=None, fname: str=None, save_flag: bool=False) -> None:

        # Perform pixel detection and scaling.
        sidd = np.abs(sicd)

        px_min = np.min(sidd)
        px_max = np.max(sidd)

        sidd = (((sidd - px_min) / (px_max - px_min)) * 255.0)

        # Generate heatmap plot.
        plt.imshow(sidd, cmap=cmap, origin=origin)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()

        if (save_flag):

            try:

                plt.savefig(os.path.join(fpath, fname))

            except (TypeError):

                print('ERROR: Filepath and filename are required to save heatmap plot.')

        plt.show()

        return
    

    # Plot confusion matrix.
    def plot_confusion_matrix(self, y: tf.Tensor, yhat: tf.Tensor, label: np.array, title: str=None) -> None:

        # Generate confusion matrix.
        cm = confusion_matrix(y_true=label[y], y_pred=label[yhat], labels=label)

        # Initialize plot parameters.
        cmap = plt.cm.Blues                     # Color map

        title = title                           # Plot title
        tick = np.arange(len(label))            # Axis labels

        fmt = 'd'                               # Data format
        thresh = (cm.max() / 2.0)               # Treshold
        
        # Plot confusion matrix.
        plt.figure(figsize=(15, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)

        plt.colorbar()
        plt.xticks(tick, label, rotation=45)
        plt.yticks(tick, label)
        
        for (i, j) in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), \
                     color=('white' if cm[i, j] > thresh else 'black'))
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.show()

        return 
    
