from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as plt


def main():
    x, y = make_blobs(cluster_std=0.5,
                      center_box=(-5, 5),
                      random_state=1)
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    markers = ['o', '^', 'x']
    colors = ['blue', 'red', 'green']
    for label, marker, color in zip(np.unique(y), markers, colors):
        plt.scatter(x[y == label, 0], x[y == label, 1],
                    marker=marker, color=color)
    plt.show()
    return 0


if __name__ == '__main__':
    main()
