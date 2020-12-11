import cupy as cp

from .layer import Layer

class MaxPool2D(Layer):

    def __init__(self,
                 kernel_shape,
                 padding='valid',
                 ):

        super().__init__(passive=True)
        self.kernel_shape = kernel_shape
        self.stride = kernel_shape[0]
        self._cache = {}

    def build(self, input_shape):
        self.input_shape = input_shape

        self.output_shape = self.max2d_output_shape(self.input_shape)
        return self.output_shape

    def forward(self, X, train=False):
        self._cache = {}
        self.batch_size = X.shape[0]
        self.X = cp.array(X, copy=train)

        n, _, _, c = X.shape
        h_out, w_out, _ = self.output_shape
        h_pool, w_pool = self.kernel_shape
        MAX = cp.zeros((n, h_out, w_out, c))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                a_prev_slice = X[:, h_start:h_end, w_start:w_end, :]
                self.save_max_mask(X=a_prev_slice, cords=(i, j))
                MAX[:, i, j, :] = cp.max(a_prev_slice, axis=(1, 2))
        return MAX


    def backward(self, dLdA, y=None):
        output = cp.zeros_like(self.X)
        _, h_out, w_out, _ = dLdA.shape
        h_pool, w_pool = self.kernel_shape

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                output[:, h_start:h_end, w_start:w_end, :] += \
                    dLdA[:, i:i + 1, j:j + 1, :] * self._cache[(i, j)]
        return output



    def save_max_mask(self, X, cords):
        mask = cp.zeros_like(X)
        n, h, w, c = X.shape
        X = X.reshape(n, h * w, c)
        idx = cp.argmax(X, axis=1)

        n_idx, c_idx = cp.indices((n, c))
    
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self._cache[cords] = mask



    def max2d_output_shape(self, input_shape):
        out_shape = [
            self.max2d_output_len(in_len, f_len)
            for in_len, f_len in zip(input_shape, self.kernel_shape)
        ]
        out_shape.append(input_shape[-1])
        return out_shape

    def max2d_output_len(self, input_len, kernel_size):
        out_len = input_len//kernel_size
        return out_len

    def clear_cache(self):
        del self._cache
        self.X = None
