from wine import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from use_colormap import plot_decision_regions


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
        scat = scat.reshape(d, 1)
        S_B += n * scat.dot(scat.T)

    # calculate eigen values,vectors of inv(S_B).dot(S_W)
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    # plot(np.abs(eigen_vals))

    # make matrix
    orderd_vecs = np.array([eigen_vecs[:, i]
                            for i in np.argsort(np.abs(eigen_vals))[::-1]])
    w = np.vstack((orderd_vecs[0].real, orderd_vecs[1].real)).T
    x_train_lda = x_train_std.dot(w)
    print(x_train_lda.shape)
    lr = LogisticRegression()
    lr.fit(x_train_lda, y_train)
    x_test_lda = x_test_std.dot(w)
    plot_decision_regions(x_train_lda, y_train, classifier=lr)
    plot_decision_regions(x_test_lda, y_test, classifier=lr)


def plot(vals):
    vals = np.sort(vals)[::-1]
    bars = [np.sum(vals[:i])for i in range(len(vals))]
    fig = plt.figure()
    plt.bar(range(len(vals)), vals, align='edge')
    plt.step(range(len(vals)), bars, where='pre')
    fig.show()
    plt.show()


if __name__ == '__main__':
    main()
