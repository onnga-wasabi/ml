import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering


def print_linkage_mat(data):
    dist_vec = pdist(data)
    dist_mat = squareform(dist_vec)
    #=>dist_mat = cdist(data,data)
    linkage_mat = linkage(dist_vec.ravel())
    #=>linkage_mat = linkage(data)
    linked_clusters = pd.DataFrame(
        linkage_mat,
        columns=['cluster1', 'cluster2',
                 'dsitance', 'num of clusters'],
        index=['cluster' + str(i + 1)
               for i in range(linkage_mat.shape[0])])
    print(linked_clusters)
    return linked_clusters


def show_dend(data, df):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    axd = fig.add_subplot(1, 2, 1)
    dend = dendrogram(df, orientation='left')

    axm = fig.add_subplot(1, 2, 2)
    sns.heatmap(data[dend['leaves'][::-1]])
    axd.set_xticks([])
    axm.set_yticks([])
    axm.set_xticklabels(['x', 'y', 'z'])
    plt.show()

    return 0


def agglomeration(data):
    ac = AgglomerativeClustering(n_clusters=2,
                                 affinity='euclidean',
                                 linkage='complete')
    pred = ac.fit_predict(data)
    print(pred)

    return 0


def main():
    data = np.random.rand(5, 3) * 10
    df = print_linkage_mat(data)
    show_dend(data, df)
    agglomeration(data)

    return 0


if __name__ == '__main__':
    main()
