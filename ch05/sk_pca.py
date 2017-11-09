from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from wine import *
from sklearn.preprocessing import StandardScaler

from use_colormap import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt


def main():
    x_train, x_test, y_train, y_test = load_wine()
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)

    lr = LogisticRegression()
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train_std)
    x_test_pca = pca.transform(x_test_std)

    lr.fit(x_train_pca, y_train)
    plot_decision_regions(x=x_test_pca, y=y_test, classifier=lr)


if __name__ == '__main__':
    main()
