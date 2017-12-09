from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt


def valification(x, y, clfs):
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    fig.canvas.manager.window.attributes('-topmost', 1)
    ax[0, 0].scatter(x[y == 0, 0], x[y == 0, 1], color='blue')
    ax[0, 0].scatter(x[y == 1, 0], x[y == 1, 1], color='red')
    ax[0, 0].set_title('ORIGINAL')

    y_pred = clfs[0].fit_predict(x)
    ax[0, 1].scatter(x[y_pred == 0, 0],
                     x[y_pred == 0, 1], color='blue')
    ax[0, 1].scatter(x[y_pred == 1, 0],
                     x[y_pred == 1, 1], color='red')
    ax[0, 1].set_title('KMeans')

    y_pred = clfs[1].fit_predict(x)
    ax[1, 0].scatter(x[y_pred == 0, 0],
                     x[y_pred == 0, 1], color='blue')
    ax[1, 0].scatter(x[y_pred == 1, 0],
                     x[y_pred == 1, 1], color='red')
    ax[1, 0].set_title('Agglomeration')

    y_pred = clfs[2].fit_predict(x)
    ax[1, 1].scatter(x[y_pred == 0, 0],
                     x[y_pred == 0, 1], color='blue')
    ax[1, 1].scatter(x[y_pred == 1, 0],
                     x[y_pred == 1, 1], color='red')
    ax[1, 1].set_title('DBSCAN')
    plt.show()

    return 0


def main():
    x, y = make_moons(noise=0.05)
    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=100)
    ac = AgglomerativeClustering(n_clusters=2,
                                 affinity='euclidean',
                                 linkage='complete')
    db = DBSCAN(eps=0.3,
                min_samples=5,
                metric='euclidean',
                n_jobs=-1)
    clfs = [km, ac, db]
    valification(x, y, clfs)

    return 0


if __name__ == '__main__':
    main()
