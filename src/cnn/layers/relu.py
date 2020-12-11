

import cupy as cp

from .layer import Layer
from cnn.utils.functions import LeakyReLU


class _ReLU_(Layer):

    def __init__(self, alpha=0.01):
        super().__init__(passive=True)
        self.relu = LeakyReLU(alpha)

    def build(self, input_shape, train=False):
        return input_shape

    def forward(self, X, train=False):
        self.X = cp.array(X, copy=False)
        return self.relu.f(X)

    def backward(self, dLdA, y=None):
        return self.relu.df(self.X, dLdA)

    def clear_cache(self):
        self.X = None
