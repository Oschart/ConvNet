import cupy as cp

from .layer import Layer



class Dense(Layer):

    def __init__(self,
                 fan_out,
                 learning_rate=0.001,
                 lr_decay=0.9999
                 ):

        super().__init__(learning_rate=learning_rate, lr_decay=lr_decay)
        self.fan_out = fan_out

    def build(self, input_shape):
        self.input_shape = input_shape

        fan_in = self.input_shape[0]
        layer_shape = [self.fan_out, fan_in]
        return super().init_weights(layer_shape, fan_in, num_of_biases=self.fan_out)

    def forward(self, X, train=False):
        self.batch_size = X.shape[0]
        self.X = cp.array(X, copy=False)
        self.Z = cp.dot(X, self.W.T)

        self.A = self.Z + self.b
        return self.A

    def backward(self, dLdA):
        n = self.batch_size
        dW = cp.dot(dLdA.T, self.X)/n
        db = cp.sum(dLdA, axis=0) / n
        self.update_weights(dW, db)
        dX = cp.dot(dLdA, self.W)
        return dX

    def clear_cache(self):
        self.X, self.Z, self.A = None, None, None

    
