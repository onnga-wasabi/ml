from wine import *
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt


def main():
    x_train, x_test, y_train, y_test = load_wine()
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)

    # calculate scatter within class
    S_W = np.sum([np.cov(x_train_std[y_train == label].T)
                  for label in np.unique(y_train)], axis=0)

    # calculate scatter between classes
    # calculate mean vector
    x_mean = np.array([np.mean(x_train_std[y_train == label], axis=0)
                       for label in np.unique(y_train)])

    d = x_train_std.shape[1]
    S_B = np.zeros((d, d))
    mean_all = np.mean(x_train_std, axis=0)
    for label in np.unique(y_train):
        n = x_train_std[y_train == label].shape[0]
        scat = x_mean[label - 1] - mean_all
        S_B += n * scat.dot(scat.T)


if __name__ == '__main__':
    main()
