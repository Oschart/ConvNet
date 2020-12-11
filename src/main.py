# %%
import os
import pathlib
import pickle
import random
import warnings
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.model_selection import train_test_split

from cnn.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D, Rescaling,
                        _ReLU_, _SoftMax_)
from cnn.neural_network import NeuralNetwork
from cnn.utils.functions import SoftmaxCrossEntropyLoss
from cnn_utils import (load_dataset, load_model, make_dirs,
                       plot_loss_vs_epochs, plot_tuning_results, print_scores,
                       save_arrays, save_model)

warnings.filterwarnings('ignore')


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = 'flower_photos'
# check if dataset folder exists (to save download time)
if os.path.isdir(data_dir) is False:
    tf.keras.utils.get_file(origin=dataset_url,
                            fname='flower_photos',
                            untar=True)

# Image parameters
IMG_HEIGHT = 50
IMG_WIDTH = 50
USE_CACHED_DATASET = True
USE_CACHED_MODEL = True
TRAIN_MODEL = False
SEED = 1998
RUN_EXP = False

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
cp.random.seed(SEED)
np.random.seed(SEED)

as_gray = False

cache_dir = 'cache'
model_dir = '%s/model/' % cache_dir
arrays_dir = '%s/dataset_arrays/' % cache_dir
plot_dir = 'plots/'

make_dirs(cache_dir, model_dir, arrays_dir, plot_dir)

if USE_CACHED_DATASET and os.path.isfile('%s/Xtr.npy' % arrays_dir):
    Xtr, ytr = cp.load('%s/Xtr.npy' %
                       arrays_dir), cp.load('%s/ytr.npy' % arrays_dir)
    Xtest, ytest = cp.load('%s/Xtest.npy' %
                           arrays_dir), cp.load('%s/ytest.npy' % arrays_dir)
    categories = np.load('%s/categories.npy' % arrays_dir)
else:
    Xtr, ytr, Xtest, ytest, categories = load_dataset(
        data_dir, (IMG_HEIGHT, IMG_WIDTH), as_gray=as_gray)
    save_arrays(arrays_dir, Xtr=Xtr, ytr=ytr, Xtest=Xtest,
                ytest=ytest, categories=categories)
print('Dataset Loading Complete.')

X_train, X_val, y_train, y_val = train_test_split(Xtr, ytr,
                                                  test_size=0.1,
                                                  random_state=SEED)

# Final model hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 300
LOSS_T = SoftmaxCrossEntropyLoss
OPTIMIZER = 'adam'
BEST_PER_EPOCH = True
L_RATE = 0.04
BETA1 = 0.9
BETA2 = 0.99
ALPHA = 0.01
TOPOLOGY = [
    Conv2D(64, (3, 3), padding='same', learning_rate=L_RATE, lr_decay=0.99995),
    MaxPool2D((5, 5)),
    _ReLU_(ALPHA),
    Conv2D(64, (3, 3), padding='same', learning_rate=L_RATE, lr_decay=0.99995),
    MaxPool2D((5, 5)),
    _ReLU_(ALPHA),
    Conv2D(96, (3, 3), padding='same', learning_rate=L_RATE, lr_decay=0.99995),
    _ReLU_(ALPHA),
    Flatten(),
    Dropout(0.8),
    Dense(512, learning_rate=L_RATE/2, lr_decay=0.99995),
    _ReLU_(ALPHA),
    Dropout(0.7),
    Dense(512, learning_rate=L_RATE/2, lr_decay=0.99995),
    _ReLU_(ALPHA),
    Dropout(0.9),
    Dense(len(categories), learning_rate=L_RATE/2, lr_decay=0.99995),
    _SoftMax_()
]


def create_tuned_cnn():
    """Create a CNN with the standard fine-tuned parameters

    Returns:
        [NeuralNetwork]: [a fine-tuned NN]
    """
    return NeuralNetwork(
        input_shape=X_train[0].shape,
        topology=TOPOLOGY,
        X_val=X_val,
        y_val=y_val,
        learning_rate=L_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        loss=LOSS_T,
        optimizer=OPTIMIZER,
        stop_after=30
    )


def fine_tune_param(epochs, **params):
    param_name = list(params.keys())[0]
    values = params[param_name]
    val_accrs = []
    val_losses = []
    for v in values:
        cnn = create_tuned_cnn()
        cnn.max_epochs = epochs
        param = dict()
        param[param_name] = v
        cnn.set_params(**param)
        _, val_loss, val_accr = cnn.train(X_train, y_train, experimental=True)
        val_accrs.append(val_accr)
        val_losses.append(val_loss)
    return values, val_losses


def run_fine_tune_experiments():
    TUNE_EPOCHS = 20
    epochs = list(range(TUNE_EPOCHS))

    print("Start Optimizer fine-tuning experiments...")
    _, val_losses = fine_tune_param(TUNE_EPOCHS, optimizer=['basic', 'adam'])
    plot_tuning_results(['Basic', 'Adam'], val_losses, epochs, "Optimizer")

    print("Start Learning Rate fine-tuning experiments...")
    learning_rates = [0.001, 0.005, 0.01, 0.025, 0.05]
    _, val_losses = fine_tune_param(TUNE_EPOCHS, learning_rate=learning_rates)
    plot_tuning_results(learning_rates, val_losses, epochs, "Learning Rate")

    print("Start Beta1 fine-tuning experiments...")
    beta1s = [0.7, 0.8, 0.9, 0.95]
    _, val_losses = fine_tune_param(TUNE_EPOCHS, beta1=beta1s)
    plot_tuning_results(beta1s, val_losses, epochs, "Beta1")

    print("Start Beta2 fine-tuning experiments...")
    beta2s = [0.9, 0.96, 0.99, 0.999]
    _, val_losses = fine_tune_param(TUNE_EPOCHS, beta2=beta2s)
    plot_tuning_results(beta2s, val_losses, epochs, "Beta2")


# %%


def main():

    if USE_CACHED_MODEL:
        cnn = load_model('cnn', model_dir)
    else:
        cnn = create_tuned_cnn()

    if TRAIN_MODEL:
        print("Start Main Model Training...")
        cnn.set_params(batch_size=BATCH_SIZE, X_val=X_val, y_val=y_val,)
        loss_tr, loss_val, _ = cnn.train(X_train, y_train, DEBUG=True)
        num_epochs = len(loss_val)
        # save model
        save_model(cnn, 'cnn', model_dir)
        print("Main Model Finished.")
        # plot results
        plot_loss_vs_epochs(np.array(loss_tr), np.array(loss_val), list(
            range(num_epochs)), title="Loss vs. Epochs")

    Ypred = cnn.predict(Xtest)
    CRRns, ACCR = cnn.detailed_score(Ypred, ytest)
    print("Test set results:")

    print_scores(CRRns, ACCR, categories)
    if RUN_EXP:
        run_fine_tune_experiments()

    return


# %%
print('Start run for RGB mode:')
main()

# %%
