from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt
import itertools

import sys
import os
sys.path.append(os.pardir)
from utils.housing import load_housing


def show_residues(regs, labels, x_train, x_test, y_train, y_test):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.canvas.manager.window.attributes('-topmost', 1)

    for regressor, idx, title in zip(regs, itertools.product([0, 1], [0, 1]), labels):
        # train
        y_pred = regressor.predict(x_train)
        train_score = r2_score(y_train, y_pred).round(3)
        ax[idx].scatter(y_pred, y_pred - y_train, marker='^',
                        color='blue', label='train')
        # test
        y_pred = regressor.predict(x_test)
        test_score = r2_score(y_test, y_pred).round(3)
        ax[idx].scatter(y_pred, y_pred - y_test, marker='x',
                        color='green', label='test')

        ax[idx].text(0, -24, 'train r2:' + str(train_score), color='orange')
        ax[idx].text(0, -27, 'test  r2:' + str(test_score), color='orange')
        ax[idx].set_ylim([-30, 20])
        ax[idx].axhline(y=0, color='red')
        ax[idx].set_xlabel('Predicted Value[price]')
        ax[idx].set_ylabel('Residue')
        ax[idx].set_title(title)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()
    return 0


def main():
    df = load_housing('../utils/')
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # linear regression
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # lasso regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)

    # ridge regression
    lasso = Lasso(alpha=1.0)
    lasso.fit(x_train, y_train)

    # elasticnet regression
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
    # minimizes the objective function
    elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
    elastic.fit(x_train, y_train)
    labels = ['Linear Regression', 'Ridge Rgression',
              'Lasso Regression', 'Elastic Regression']
    show_residues([lr, ridge, lasso, elastic], labels,
                  x_train, x_test, y_train, y_test)
    return 0


if __name__ == '__main__':
    main()
