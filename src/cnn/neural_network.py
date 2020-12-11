# -*- coding: utf-8 -*-
# %%
from collections import Counter
from random import random, seed
import cupy as cp
import numpy as np
from tqdm import tqdm

from .layers import Layer
from cnn.utils.functions import one_hot_encode, SoftmaxCrossEntropyLoss


class NeuralNetwork(object):
    """[summary]
        Convolutional Neural Network
    """

    def __init__(self,
                 input_shape=0,
                 topology=[],
                 batch_size=32,
                 max_epochs=10,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.99,
                 loss=SoftmaxCrossEntropyLoss,
                 optimizer='adam',
                 X_val=None,
                 y_val=None,
                 best_per_epoch=True,
                 num_of_classes=5,
                 stop_after=5,
                 ):
        self.input_shape = input_shape
        self.topology = topology
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        Layer.learning_rate = learning_rate
        Layer.beta1 = beta1
        Layer.beta2 = beta2

        self.X_val = X_val
        self.y_val = y_val

        self.best_per_epoch = best_per_epoch

        self.alpha = 0.1
        self.margin = 0.1
        self.best_accr = -1
        self.min_loss = 1e10

        self.loss = loss(num_of_classes)
        self.optimizer = optimizer

        self.num_of_classes = num_of_classes
        self.stop_after = stop_after

        self.build_layers()

    def build_layers(self):
        D = len(self.topology)
        self.D = D
        self.layers = []
        fan_in_shape = self.input_shape
        for layer in self.topology:
            fan_in_shape = layer.build(input_shape=fan_in_shape)
            self.layers.append(layer)

    def save_best_state(self):
        for layer in self.layers:
            layer.save_layer_state()

    def restore_best_state(self):
        for layer in self.layers:
            layer.restore_layer_state()

    def train(self, X, y, DEBUG=False, experimental=False):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        stopping_cnter = 0
        tr_losses = []
        val_losses = []
        val_accrs = []
        # bX, bY = self.batch_sample(X, y)
        #y_hot = one_hot_encode(y, h_k=self.num_of_classes)
        npb = np.ceil(X.shape[0]/300)
        band_lim = 0.5
        for i in range(self.max_epochs):
            num_batches = np.ceil(X.shape[0]/self.batch_size)
            for bX, bY in tqdm(self.batch_sample(X, y, self.batch_size), desc="Epoch %s" % i, total=num_batches):
                self.apply_SGD_step(bX, bY)

            if DEBUG:
                accr_tr = 0.0
                loss_tr = 0.0
                for bX, bY in self.batch_sample(X, y, 300):
                    accr_tr += self.accuracy_raw(bX, bY)/npb
                    loss_tr += cp.average(self.loss.f(self.forward_pass(bX), bY))/npb
            else:
                accr_tr = '---'
                loss_tr = '---'

            accr_val = self.accuracy_raw(self.X_val, self.y_val)

            loss_val = self.loss.f(self.forward_pass(self.X_val), self.y_val)
            loss_val = cp.average(loss_val)

            tr_losses.append(loss_tr)
            val_losses.append(loss_val)
            val_accrs.append(accr_val)

            if experimental is False:
                if self.best_accr < accr_val:
                    self.best_accr = accr_val
                    self.save_best_state()
                
                if self.min_loss > loss_val:
                    stopping_cnter = 0
                    self.min_loss = loss_val
                else:
                    stopping_cnter += 1

                if accr_val > band_lim:
                    self.batch_size += 5
                    band_lim += 0.1

                # stop if validation loss doesn't improve after x epochs
                if stopping_cnter >= self.stop_after:
                    if self.best_per_epoch:
                        self.restore_best_state()
                    return tr_losses, val_losses, val_accrs

            if DEBUG:
                print('Epoch %d: train_accr = %s, val_accr = %s, avg_loss_tr = %s, avg_loss_val = %s' %
                      (i, accr_tr, accr_val, loss_tr, loss_val))

        if self.best_per_epoch:
            self.restore_best_state()

        return tr_losses, val_losses, val_accrs

    def forward_pass(self, bX, train=False):
        f_feed = bX
        for i in range(self.D):
            f_feed = self.layers[i].forward(f_feed, train=train)
        return f_feed

    def backward_pass(self, d_loss):
        b_feed = d_loss
        for i in reversed(range(self.D)):
            b_feed = self.layers[i].backward(b_feed)

    def apply_SGD_step(self, bX, bY):
        output_activ = self.forward_pass(bX, train=True)
        d_loss = self.loss.df(output_activ, bY)
        self.backward_pass(d_loss)

    def batch_sample(self, X, y, bsize):
        for i in range(0, X.shape[0], bsize):
            st, en = i, min(i + bsize, X.shape[0])
            yield X[st: en], y[st: en]

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        Ypred = cp.zeros(num_test)
        scores = self.forward_pass(X)
        Ypred = cp.argmax(scores, axis=1)

        return Ypred

    def accuracy(self, predY, y):
        return sum(predY == y)/y.shape[0]

    def accuracy_raw(self, X, y):
        predY = self.predict(X)
        return cp.sum(predY == y)/y.shape[0]

    def detailed_score(self, predY, y, num_of_class=5):
        CR = [0]*num_of_class
        TCR = [0]*num_of_class
        predY_ = cp.asnumpy(predY)
        y_ = cp.asnumpy(y)
        for pY, cY in zip(predY_, y_):
            TCR[cY] = TCR[cY] + 1
            CR[cY] = CR[cY] + (pY == cY)
        CCRn = [cr/tcr for (cr, tcr) in zip(CR, TCR)]
        self.accR = sum(CR)/sum(TCR)
        return CCRn, self.accR

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def clear_cache(self):
        self.X_val, self.y_val = None, None
        for layer in self.layers:
            layer.clear_cache()