from cancer import load_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt


def main():
    x, y = load_cancer()
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3)
    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=2)),
                        ('clf', LogisticRegression(
                            penalty='l2',
                            C=100))])
    pipe_lr.fit(x_train, y_train)
    print(pipe_lr.score(x_test, y_test))
    cv = StratifiedKFold(n_splits=3)
    x_train2 = x_train[:, [4, 14]]
    for train, test in cv.split(x_train, y_train):
        probas = pipe_lr.fit(
            x_train2[train], y_train[train]).predict_proba(x_train2[test])
        fpr, tpr, thresholds = roc_curve(
            y_train[test], probas[:, 1], pos_label=1)
        plt.plot(fpr, tpr, lw=1)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.show()


if __name__ == '__main__':
    main()
