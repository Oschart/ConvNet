import cupy as cp


class Layer():
    # Static across all layers (default)
    #learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.99
    optimizer = 'adam'
    batch_size = 32
    lr_min = 0.007

    def __init__(self,
                 passive=False,
                 learning_rate=0.01,
                 lr_decay=0.9999
                 ):
        self.passive = passive
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.W_std = 0
        self.b_std = 0

    def build(self, input_shape=None):
        print('build not supported for layer!')
        pass

    def init_weights(self, layer_shape, fan_in, num_of_biases=0):
        self.b = cp.random.randn(num_of_biases) * 0.1

        self.mom = 0
        self.acc = 0
        self.mom_b = 0
        self.acc_b = 0
        # Xavier initialization
        self.W = cp.random.randn(*layer_shape)/cp.sqrt(fan_in)
        output_shape = layer_shape[:-1]
        return output_shape

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def forward(self, A_prev, train=False):
        print('forward pass not supported for layer!')

    def backward(self, dLdA, y=None):
        print('backward pass not supported for layer!')

    def update_weights(self, dW, db):

        # Passive layer: no weights
        if self.passive:
            return

        if self.optimizer == 'basic':
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            return

        # Adam Optimizer
        self.mom = self.beta1*self.mom + (1-self.beta1)*dW
        self.acc = self.beta2*self.acc + \
            (1-self.beta2)*(dW*dW).sum()

        self.mom_b = self.beta1*self.mom_b + (1-self.beta1)*db
        self.acc_b = self.beta2*self.acc_b + \
            (1-self.beta2)*(db*db).sum()

        self.W -= self.learning_rate * \
            self.mom / (cp.sqrt(self.acc) + 1e-7)
        self.b -= self.learning_rate * \
            self.mom_b / (cp.sqrt(self.acc_b) + 1e-7)

        self.learning_rate *= self.lr_decay
        self.learning_rate = max(self.learning_rate, self.lr_min)

    def save_layer_state(self):
        if self.passive:
            return
        del self.W_std
        del self.b_std
        self.W_std = cp.array(self.W, copy=True)
        self.b_std = cp.array(self.b, copy=True)

    def restore_layer_state(self):
        if self.passive:
            return
        self.W = self.W_std
        self.b = self.b_std

    def clear_cache(self):
        pass
