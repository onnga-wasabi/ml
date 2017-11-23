from sklearn.datasets import make_moons
from rbf import rbf_kernel_pca
import matplotlib.pyplot as plt
import numpy as np


def main():
    # make data
    x, y = make_moons()

    # rbf_kernel_pca(data,gamma,n components)
    x_pca, lambdas = rbf_kernel_pca(x, 15, 1)
    #show_data(x_pca, y)
    x_new = x[25]
    print('original projected:', x_pca[25])
    x_projected = x_project(x_new, x, 15, x_pca, lambdas)
    print('new projected data:', x_projected)
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.scatter(x_pca[y == 0], np.zeros((len(x_pca[y == 0]), 1)), marker='o')
    plt.scatter(x_pca[y == 1], np.zeros((len(x_pca[y == 1]), 1)), marker='^')
    plt.scatter(x_projected, 0, marker='x', color='red')
    fig.canvas.manager.window.attributes('-topmost', 1)

    plt.pause(4)


def x_project(x_new, x, gamma, alphas, lambdas):
    dist_pairs = np.array([np.sum((x_new - row)**2) for row in x])
    K = np.exp(-gamma * dist_pairs)
    return K.dot(alphas / lambdas)


def show_data(x, y):
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.scatter(x[y == 0, 0], x[y == 0, 1], marker='o')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='^')
    fig.canvas.manager.window.attributes('-topmost', 1)

    plt.pause(2)


if __name__ == '__main__':
    main()
