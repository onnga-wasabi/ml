from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(x, gamma, n_components):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma * mat_sq_dists)
    d = K.shape[0]
    one_nth = np.ones((d, d)) / d
    K = K - one_nth.dot(K) - K.dot(one_nth) + one_nth.dot(K).dot(one_nth)
    eigen_vals, eigen_vecs = eigh(K)
    print('eigen_vecs\'s shape:', eigen_vecs.shape)

    # 上位の固有ベクトルを取得
    alphas = eigen_vecs[:, -1:-n_components - 1:-1]

    lambdas = [eigen_vals[-i] for i in range(1, n_components + 1)]
    return alphas, lambdas
