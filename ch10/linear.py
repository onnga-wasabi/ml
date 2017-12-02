import numpy as np


class LinearRegressionGD(object):
    def __init__(self, n_iter=20, eta=0.001):
        self.n_iter = n_iter
        self.eta = eta

    def fit(self, x, y):
        self.costs_ = []
        dim = x.shape[1]
        self.weights_ = np.zeros(1 + dim)
        for _ in range(self.n_iter):
            errors = y - self.predict(x)
            self.weights_[1:] += self.eta * x.T.dot(errors)
            self.weights_[0] += self.eta * np.sum(errors)
            self.costs_.append(np.sum(errors**2) / 2)
        return self

    def predict(self, x):
        return np.dot(x, self.weights_[1:]) + self.weights_[0]
