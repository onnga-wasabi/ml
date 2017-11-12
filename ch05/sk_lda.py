from wine import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import numpy as np
import matplotlib.pyplot as plt
from use_colormap import plot_decision_regions


def main():
    x_train, x_test, y_train, y_test = load_wine()
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)

    lda = LDA(n_components=2)
    x_train_lda = lda.fit_transform(x_train_std, y_train)
    x_test_lda = lda.transform(x_test_std)

    lr = LogisticRegression()
    lr.fit(x_train_lda, y_train)

    plot_decision_regions(x_train_lda, y_train, classifier=lr)
    plot_decision_regions(x_test_lda, y_test, classifier=lr)


if __name__ == '__main__':
    main()
