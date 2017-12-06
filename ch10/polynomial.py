from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.pardir)
from utils.housing import load_housing


def show_regression(lr, X, y, xa):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    plt.scatter(X[0], y, marker='x', color='pink')
    for i, x in enumerate(X):
        lr.fit(X[i], y)
        y_pred = lr.predict(xa[i])
        r2 = r2_score(y, lr.predict(X[i])).round(2)
        plt.plot(xa[0], y_pred, label='dim=' +
                 str(i + 1) + 'R2 score:' + str(r2))
    plt.xlabel('lower status of the population%')
    plt.ylabel('median value of homes in $1000\'s')
    plt.legend()
    plt.show()
    return 0


def main():
    df = load_housing('../utils/')
    x = df[['LSTAT']].values
    y = df["MEDV"].values
    pl2 = PolynomialFeatures(degree=2)
    pl3 = PolynomialFeatures(degree=3)
    x2 = pl2.fit_transform(x)
    x3 = pl3.fit_transform(x)
    lr = LinearRegression()
    X = [x, x2, x3]
    xa = []
    xa.append(np.arange(X[0].min(), X[0].max(), 1).reshape(-1, 1))
    xa.append(pl2.fit_transform(xa[0]))
    xa.append(pl3.fit_transform(xa[0]))
    show_regression(lr, X, y, xa)
    return 0


if __name__ == '__main__':
    main()