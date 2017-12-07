from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt


def show_data(x, y):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    fig.canvas.manager.window.attributes('-topmost', 1)
    markers = ['o', '^', 'x']
    colors = ['blue', 'red', 'green']
    for label, marker, color in zip(np.unique(y), markers, colors):
        ax1.scatter(x[y == label, 0], x[y == label, 1],
                    marker=marker, color=color)
    ax2.scatter(x[:, 0], x[:, 1], marker='8')
    ax1.set_title('correct label')
    ax2.set_title('masked correct label')
    plt.show()
    return 0


def verification(x, y, y_pred):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    fig.canvas.manager.window.attributes('-topmost', 1)
    markers = ['o', '^', 'x']
    colors = ['blue', 'red', 'green']
    for label, marker, color in zip(np.unique(y), markers, colors):
        ax1.scatter(x[y == label, 0], x[y == label, 1],
                    marker=marker, color=color)
        ax2.scatter(x[y_pred == label, 0], x[y_pred == label, 1],
                    marker=marker, color=color)
    ax1.set_title('correct label')
    ax2.set_title('masked correct label')
    plt.show()
    return 0


def main():
    x, y = make_blobs(cluster_std=0.5,
                      center_box=(-5, 5),
                      random_state=1)
    #show_data(x, y)
    # k=n_cluster:クラスタの数
    km = KMeans(n_clusters=3,
                n_init=10,
                max_iter=300,
                tol=1e-4,
                n_jobs=-1)
    y_pred = km.fit_predict(x, y)
    verification(x, y, y_pred)

    return 0


if __name__ == '__main__':
    main()
