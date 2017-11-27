from cancer import load_cancer
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

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
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    show_confusion_matrix(confmat)


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
