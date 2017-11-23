from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt


def main():
    x, y = make_moons()
    scikit_kpca = KernelPCA(n_components=2, gamma=15, kernel='rbf')
    x_kernelpca = scikit_kpca.fit_transform(x)
    fig = plt.figure()
    plt.scatter(x_kernelpca[y == 0, 0],
                x_kernelpca[y == 0, 1], marker='o', color='blue')
    plt.scatter(x_kernelpca[y == 1, 0],
                x_kernelpca[y == 1, 1], marker='^', color='yellow')
    fig.canvas.manager.window.attributes('-topmost', 1)

    plt.pause(3)


if __name__ == '__main__':
    main()
