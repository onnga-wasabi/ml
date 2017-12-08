from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

import numpy as np
import matplotlib.pyplot as plt


def verification(km, x, y):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    fig.canvas.manager.window.attributes('-topmost', 1)
    markers = ['o', '^', 'x']
    colors = ['blue', 'orange', 'green']
    y_pred = km.fit_predict(x)
    for label, marker, color in zip(np.unique(y), markers, colors):
        ax1.scatter(x[y == label, 0], x[y == label, 1],
                    marker=marker, color=color)
        ax2.scatter(x[y_pred == label, 0], x[y_pred == label, 1],
                    marker=marker, color=color)
    ax2.scatter(km.cluster_centers_[:, 0],
                km.cluster_centers_[:, 1],
                marker='*',
                s=100,
                color='red',
                label='cluster centers')
    ax1.set_title('correct label')
    ax2.set_title('predicted classes by k-means++')
    plt.legend()
    plt.show()
    return 0


def elbow(x):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    inertias = []
    num_of_clusters = np.arange(1, 11)
    for i in num_of_clusters:
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    n_jobs=-1)
        km.fit(x)
        # sum of squred distance of samples to their closest cluster center
        inertias.append(km.inertia_)
    plt.plot(num_of_clusters, inertias, marker='o')
    plt.ylabel('Distortion')
    plt.xlabel('Clusters')
    plt.show()

    return 0


def silhouette(km, x):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    y_pred = km.fit_predict(x)
    # may be mean score
    #coefficient = silhouette_score(x, y_pred,metric='euclidean')

    # return Silhouette Coefficient for each samples.
    coefficients = silhouette_samples(x, y_pred)

    return 0


def main():
    x, y = make_blobs(cluster_std=0.5,
                      center_box=(-5, 5),
                      random_state=1)
    #show_data(x, y)
    # k=n_cluster:クラスタの数
    km = KMeans(n_clusters=3,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-4,
                n_jobs=-1)
    #verification(km, x, y)
    # elbow(x)
    silhouette(km, x)

    return 0


if __name__ == '__main__':
    main()
