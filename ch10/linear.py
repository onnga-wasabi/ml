import numpy as np


class LinearRegressionGD(object):
    def __init__(self, n_iter=50, eta=0.001):
        self.n_iter = n_iter
        self.eta = eta

    def fit(self, x, y):
        self.costs_ = []
        self.weights_ = np.zeros(1 + x.shape[1])
        for _ in range(self.n_iter):
            errors = y - self.predict(x)
            self.costs_.append(np.sum(0.5 * (errors**2)))
            self.weights_[1:] += self.eta * x.T.dot(errors)
            self.weights_[0] += self.eta * np.sum(errors)
        return self

    def predict(self, x):
        return x.dot(self.weights_[1:]) + self.weights_[0]
