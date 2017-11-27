from cancer import load_cancer
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt


def main():
    x, y = load_cancer()

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.2)

    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC())])
    pipe_svc.fit(X=x_train, y=y_train)
    y_pred = pipe_svc.predict(x_test)
    # show_confusion_matrix(confmat)
    print('precision score:', precision_score(y_true=y_test, y_pred=y_pred))
    print('recall score:', recall_score(y_true=y_test, y_pred=y_pred))
    print('f1 score:', f1_score(y_true=y_test, y_pred=y_pred))

    # gridsearchのデフォルトでは陽性クラスがラベル１で設定されているため、
    # ラベルを変更するにはmake_scorerで自作の性能評価関数を定義してやれば良い。
    param_range = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                  {'clf__C': param_range, 'clf__gamma': param_range,
                      'clf__kernel': ['rbf']}]
    scorer = make_scorer(f1_score, pos_label=0)
    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring=scorer,
        cv=10)
    gs.fit(x_train, y_train)
    es = gs.best_estimator_
    es.fit(x_train, y_train)
    print(es.score(x_test, y_test))


def show_confusion_matrix(confmat):
    fig = plt.figure()
    plt.matshow(confmat, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            plt.text(x=j, y=i, s=confmat[i, j])
    fig.canvas.manager.window.attributes('-topmost', 1)
    plt.pause(2)


if __name__ == '__main__':
    main()
