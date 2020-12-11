

import cupy as cp

from .layer import Layer
from cnn.utils.functions import SoftMax

class _SoftMax_(Layer):

    def __init__(self):
        super().__init__(passive=True)
        self.softmax = SoftMax()



    def build(self, input_shape):
        return input_shape

    def forward(self, X, train=False):
        self.A = self.softmax.f(X)
        return self.A

    def backward(self, dLdA, y=None):
        return dLdA

