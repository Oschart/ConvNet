import cupy as cp
# from cupyx.scipy.signal import convolve2d
from scipy.signal import convolve2d, correlate2d
import numpy as np

from .layer import Layer
from scipy import signal
from ..utils.conv2d_utils import im2col, col2im


class Conv2D(Layer):

    def __init__(self,
                 N_K,
                 kernel_shape,
                 padding='valid',
                 learning_rate=0.01,
                 lr_decay=0.9999
                 ):

        super().__init__(learning_rate=learning_rate, lr_decay=lr_decay)
        self.N_K = N_K
        self.kernel_shape = kernel_shape
        self.padding = padding

    def build(self, input_shape):
        self.input_shape = input_shape
        W = self.kernel_shape[0]
        H = self.kernel_shape[1]
        D = self.input_shape[2]
        # Layer weights construction
        self.kernel_shape = (W, H, D)

        layer_shape = (W, H, D, self.N_K)
        fan_in = np.dot(self.input_shape, self.input_shape)
        super().init_weights(layer_shape, fan_in, num_of_biases=self.N_K)

        # Handle batching
        self.output_shape = self.conv_output_shape(self.input_shape)
        self.pad = self.conv_pad(W)
        self.window_shape = self.output_shape[:2]
        return self.output_shape

    def forward(self, X, train=False):
        self.batch_size = X.shape[0]
        self.X = cp.array(X, copy=train)

        self.Z = self.convolve2d(X)
        self.A = self.Z.transpose(3, 1, 2, 0) + self.b
        return self.A

    def backward(self, dLdA, y=None):
        n = self.batch_size
        db = dLdA.sum(axis=(0, 1, 2)) / n

        dLdA_r = dLdA.transpose(3, 1, 2, 0).reshape(self.N_K, -1)
        w = cp.transpose(self.W, (3, 2, 0, 1))
        dW = dLdA_r.dot(self._cols.T).reshape(w.shape)
        dW = cp.transpose(dW, (2, 3, 1, 0))

        output_cols = w.reshape(self.N_K, -1).T.dot(dLdA_r)
        output = col2im(
            cols=output_cols,
            array_shape=cp.moveaxis(self.X, -1, 1).shape,
            filter_dim=self.kernel_shape[:2],
            pad=self.pad,
            stride=1
        )
        self.update_weights(dW, db)
        return cp.transpose(output, (0, 2, 3, 1))

    def convolve2d(self, X):
        w = cp.transpose(self.W, (3, 2, 0, 1))
        out_dim = self.output_shape[:2]
        self._cols = im2col(
            array=cp.moveaxis(X, -1, 1),
            filter_dim=self.kernel_shape[:2],
            pad=self.pad,
            stride=1
        )
        result = w.reshape((self.N_K, -1)).dot(self._cols)
        conv_out = result.reshape(
            self.N_K, out_dim[0], out_dim[1], self.batch_size)
        return conv_out

    def conv_output_shape(self, input_shape):
        out_shape = [
            self.conv_output_len(in_len, f_len)
            for in_len, f_len in zip(input_shape, self.kernel_shape)
        ]
        out_shape[-1] = self.N_K
        return out_shape

    def conv_output_len(self, input_len, kernel_size):
        if self.padding == 'valid':
            return input_len - kernel_size + 1
        return input_len

    def conv_pad(self, kernel_dim):
        if self.padding == 'valid':
            return 0
        return (kernel_dim - 1) // 2

    def clear_cache(self):
        self.X, self.Z, self.A, self._cols = None, None, None, None
