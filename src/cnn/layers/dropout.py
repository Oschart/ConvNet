import cupy as cp

from .layer import Layer

class Dropout(Layer):

    def __init__(self,
                 p,
                 learning_rate=0.001,
                 lr_decay=0.9999
                 ):

        super().__init__(passive=True, learning_rate=learning_rate, lr_decay=lr_decay)
        assert 0 < p < 1, "Probability must be between 0 and 1"
        self.p = p
        self.mask_dim = None
        self.mask = None

    def build(self, input_shape):
        self.input_shape = input_shape
        self.mask_dim = input_shape
        return input_shape

    def forward(self, X, train=False):
        if train:
            mask = (cp.random.rand(*X.shape) < self.p)
            a = self.dropout_grad(X, mask)

            # Save for backward pass
            self.mask = mask

            return a

        return X

    def backward(self, dLdA):
        return self.dropout_grad(dLdA, self.mask)
    
    def dropout_grad(self, a, mask):
        a *= mask
        a /= self.p
        return a
    
