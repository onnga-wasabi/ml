from sklearn.datasets import make_circles
from rbf import rbf_kernel_pca
import matplotlib.pyplot as plt
import numpy as np


def main():
    x, y = make_circles(n_samples=1000, noise=0.1, factor=.1)
    #show_data(x, y)
    x_pca = rbf_kernel_pca(x, 1, 2)
    show_data(x_pca, y)
    show_PC1(x_pca, y)


def show_data(x, y):
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.scatter(x[y == 0, 0], x[y == 0, 1], marker='^', color='blue')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='o', color='yellow')
    fig.canvas.manager.window.attributes('-topmost', 1)

    plt.pause(2)


def show_PC1(x, y):
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.scatter(x[y == 0, 0], np.zeros(
        (len(x[y == 0, 1]), 1)), marker='^', color='blue')
    plt.scatter(x[y == 1, 0], np.zeros((len(x[y == 1, 1]), 1)),
                marker='o', color='yellow')
    fig.canvas.manager.window.attributes('-topmost', 1)

    plt.pause(2)


if __name__ == '__main__':
    main()
