from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.pardir)
from utils.housing import load_housing


def load_data(test_size=0.3):
    df = load_housing('../utils/')
    # xを縦ベクトルにする
    x = df[['RM']].values
    y = df['MEDV'].values

    return train_test_split(x, y, test_size=test_size)


def show_regressionline(lr, x, y):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    plt.scatter(x, y, color='blue', marker='x')
    plt.plot(x, lr.predict(x), color='red')
    plt.pause(1)

    return 0


def main():
    x_train, x_test, y_train, y_test = load_data()
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print('slope    :', lr.coef_[0])
    print('intercept:', lr.intercept_)
    show_regressionline(lr, x_test, y_test)

    return 0


if __name__ == '__main__':
    main()
