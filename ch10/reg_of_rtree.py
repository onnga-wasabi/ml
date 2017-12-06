from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.pardir)
from utils.housing import load_housing


def show_redisue(rfr, x_train, x_test, y_train, y_test):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    train_y_pred = rfr.predict(x_train)
    score = r2_score(y_train, train_y_pred).round(2)
    plt.scatter(train_y_pred, train_y_pred - y_train, marker='^', s=5,
                color='orange', label='train data R2 score: ' + str(score))
    test_y_pred = rfr.predict(x_test)
    score = r2_score(y_test, test_y_pred).round(2)
    plt.scatter(test_y_pred, test_y_pred - y_test, marker='o', s=5,
                color='green', label='test data R2 score: ' + str(score))
    plt.axhline(y=0, color='red')
    plt.legend()
    plt.show()
    return 0


def main():
    df = load_housing('../utils/')
    x = df.iloc[:, :-1].values
    y = df["MEDV"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    rfr = RandomForestRegressor(n_estimators=10,
                                n_jobs=-1)
    rfr.fit(x_train, y_train)
    show_redisue(rfr, x_train, x_test, y_train, y_test)
    return 0


if __name__ == '__main__':
    main()
