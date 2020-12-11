import cupy as cp

class SoftMax():
    def f(self, X):
        A_exp = cp.exp(X - X.max(axis=1, keepdims=True))
        out = A_exp / cp.sum(A_exp, axis=1, keepdims=True)
        return out
    
    def df(self, A):
        dA = cp.ndarray(shape=A.shape)
        for bidx in range(A.shape[0]):
            s = A[bidx, :].reshape(-1,1)
            dA[bidx, :] = cp.sum(cp.diagflat(s) - cp.dot(s, s.T), axis=0)
        return dA

class LeakyReLU():
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def f(self, X):
        return cp.maximum(X, self.alpha*X)
    
    def df(self, X, dLdA):
        dX = cp.array(dLdA, copy=True)
        dX[X <= 0] *= self.alpha
        return dX


class SoftmaxCrossEntropyLoss():
    def __init__(self, n_c=5):
        self.n_c = n_c

    def f(self, S, y):
        eps = 1e-18
        y_hot = one_hot_encode(y, self.n_c)
        n = y.shape[0]
        loss = - cp.sum(y_hot * cp.log(cp.clip(S, eps, 1.))) / n
        return loss
    
    def df(self, S, y):
        #eps = 1e-15
        #y_hot = one_hot_encode(y, self.n_c)
        #return - cp.divide(y_hot, cp.clip(S, eps, 1.0))
        Sm = cp.array(S, copy=True)
        Sm[list(range(y.shape[0])), y] -= 1
        return Sm


def one_hot_encode(y, h_k):
    return cp.eye(h_k)[y]



