from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.pardir)
from utils.housing import load_housing


def show_regressionline(dtr, x, y):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    plt.scatter(x, y, marker='x', color='blue', label='sample data')
    xa = np.arange(x.min(), x.max(), 0.1).reshape(-1, 1)
    y_pred = dtr.predict(xa)
    plt.plot(xa, y_pred, color='red', linewidth=2, label='Decision Line')
    plt.legend()
    plt.show()
    return 0


def main():
    df = load_housing('../utils/')
    x = df[['LSTAT']].values
    y = df["MEDV"].values
    dtr = DecisionTreeRegressor(max_depth=3)
    dtr.fit(x, y)
    show_regressionline(dtr, x, y)
    return 0


if __name__ == '__main__':
    main()
