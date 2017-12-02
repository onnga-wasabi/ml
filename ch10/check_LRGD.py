from linear import LinearRegressionGD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.pardir)
from utils.wine import load_wine


def show_costs(clf):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)
    data = np.array([[i, cost] for i, cost in enumerate(clf.costs_)]).T
    plt.plot(data[0, :], data[1, :])
    plt.xlabel('epoch')
    plt.ylabel('costs')
    # plt.show()
    plt.pause(0.5)


def main():
    x_train, x_test, y_train, y_test = load_wine()
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)
    lr = LinearRegressionGD()
    lr.fit(x_train_std, y_train)
    show_costs(lr)
    print('R^2 score:', r2_score(y_test, lr.predict(x_test_std)).round(3))

    return 0


if __name__ == '__main__':
    main()
