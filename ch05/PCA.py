from wine import *
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt


def main():
    x_train, x_test, y_train, y_test = load_wine()
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)

    # shapeを見れば一目瞭然だが、各成分の類似度を使いたいので計算の都合上、転置
    cov_mat = np.cov(x_train_std.T)
    # print(cov_mat.shape)

    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    # make_dict
    orderd_eigen_vecs = np.array([list(eigen_vecs[:, i])
                                  for i in np.abs(eigen_vals).argsort()[::-1]])
    w = np.array([np.hstack([i, j])
                  for i, j in zip(orderd_eigen_vecs[0], orderd_eigen_vecs[1])])

    x_train_pca = x_train_std.dot(w)
    plot_pca(x_train_pca, y_train)


def plot_pca(x_train_pca, y_train):
    # pcaによって得られた2軸に注目してデータを見た時クラスごとに分布していれば効果ありだよね？
    fig = plt.figure()
    color = ['red', 'blue', 'green']
    marker = ['o', 'x', 's']
    for l, c, m in zip(np.unique(y_train), color, marker):
        plt.scatter(x_train_pca[y_train == l, 0],
                    x_train_pca[y_train == l, 1], label=l, c=c, marker=m)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    fig.show()
    plt.show()


def plot_eigen_vals(eigen_vals):
    print(eigen_vals)
    total = sum(eigen_vals)
    var_explain = [i / total for i in sorted(eigen_vals, reverse=True)]

    # np.cumsum():分散説明率の累積和
    cum_var_explain = np.cumsum(var_explain)

    fig = plt.figure()
    plt.bar(range(len(eigen_vals)), var_explain, label='explained variance')
    plt.step(range(len(eigen_vals)), cum_var_explain,
             where='mid', label='cumulative explained variance')
    plt.ylim(0, 1)
    plt.legend()
    fig.show()
    plt.show()


if __name__ == '__main__':
    main()
