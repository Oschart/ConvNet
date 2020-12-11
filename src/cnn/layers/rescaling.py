import cupy as cp

from .layer import Layer

class Rescaling(Layer):

    def __init__(self,
                 rescale_fact,
                 input_shape=None,
                 ):

        super().__init__(padding=None, passive=True)
        self.rescale_fact = rescale_fact
        self.input_shape = input_shape

    def build(self, input_shape=None):
        self.input_shape = input_shape
        return self.input_shape

    def forward(self, X):
        self.batch_size = X.shape[0]
        self.X = X
        self.A = X*self.rescale_fact

        return self.A

    def backward(self, dLdA, y=None):
        dX = dLdA*self.rescale_fact
        return dX


# %%
