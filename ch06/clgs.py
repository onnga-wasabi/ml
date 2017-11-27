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
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

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
                      cv=2,
                      n_jobs=-1)
    scores = cross_val_score(
        estimator=gs,
        X=x_train,
        y=y_train,
        cv=5)
    print('svm --gridsearch\n', scores)
    print(np.mean(scores), '+/-', np.std(scores))
    print('\n')

    gs = GridSearchCV(estimator=DecisionTreeClassifier(),
                      param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                      cv=2,
                      n_jobs=-1)
    scores = cross_val_score(
        estimator=gs,
        X=x_train,
        y=y_train,
        cv=5)
    print('DecisionTree --gridsearch\n', scores)
    print(np.mean(scores), '+/-', np.std(scores))


if __name__ == '__main__':
    main()
