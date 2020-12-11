import cupy as cp
import numpy as np


from .layer import Layer

class Flatten(Layer):

    def __init__(self):

        super().__init__(passive=True)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.fan_out = np.prod(self.input_shape)
        return [self.fan_out]

    def forward(self, IN, train=False):
        self.batch_size = IN.shape[0]
        self.IN = IN
        self.OUT = IN.reshape(self.batch_size, -1)
        return self.OUT

    def backward(self, dLdA, y=None):
        return dLdA.reshape(self.batch_size, *self.input_shape)
