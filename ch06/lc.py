from cancer import load_cancer
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt


def main():
    x, y = load_cancer()

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.2)

    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('clr', LogisticRegression())])

    # learning curve returns (n_unique_ticks),array(n_ticks,n_cv_folds)x2
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipe_lr,
        X=x_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1, 10),
        cv=10,
        n_jobs=-1)
    # train_sizeはk回の学習の中のそれぞれで使うデータ数を指定できる
    # ここで指定したデータ数に応じてtrain_sizeも返される
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    show_learning_curve(
        train_sizes,
        train_mean,
        train_std,
        test_mean,
        test_std)


def show_learning_curve(
        train_sizes,
        train_mean,
        train_std,
        test_mean,
        test_std):
    fig = plt.figure()

    # show train_accuracy
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o', label='train accuracy')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.2,
                     color='blue')

    # show validation accuracy
    plt.plot(train_sizes, test_mean,
             color='green', marker='^', label='validation accuracy')
    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.2,
                     color='green')
    fig.canvas.manager.window.attributes('-topmost', 1)
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.show()
    # plt.pause(2)


if __name__ == '__main__':
    main()
