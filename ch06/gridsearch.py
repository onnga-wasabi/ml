from cancer import load_cancer
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt


def main():
    x, y = load_cancer()

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.2)

    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC())])
    param_range = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                  {'clf__C': param_range, 'clf__gamma': param_range,
                      'clf__kernel': ['rbf']}]
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      cv=10,
                      n_jobs=-1)
    gs.fit(x_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)
    clf = gs.best_estimator_
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))


if __name__ == '__main__':
    main()
