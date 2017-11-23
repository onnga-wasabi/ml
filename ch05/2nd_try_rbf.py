from sklearn.datasets import make_moons
from rbf import rbf_kernel_pca
import matplotlib.pyplot as plt
import numpy as np


def main():
    # make data
    x, y = make_moons()

    # rbf_kernel_pca(data,gamma,n components)
    x_pca = rbf_kernel_pca(x, 15, 2)
    show_data(x_pca, y)


def show_data(x, y):
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.scatter(x[y == 0, 0], np.zeros((len(x[y==0,0]),1)), marker='o')
    plt.scatter(x[y == 1, 0], np.zeros((len(x[y==1,0]),1)), marker='^')
    fig.canvas.manager.window.attributes('-topmost', 1)

    plt.pause(4)


if __name__ == '__main__':
    main()
