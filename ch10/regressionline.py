from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

from linear import LinearRegressionGD
import os
import sys
sys.path.append(os.pardir)
from utils.housing import load_housing


def make_data(test_size=0.3):
    df = load_housing('../utils/')
    df = df[['RM', 'MEDV']]
    y = df[['MEDV']].values
    x = df[['RM']].values
    sc = StandardScaler()
    x_std = sc.fit_transform(x)
    # 識別器にぶっこむときに目的変数は1次元を想定
    y_std = sc.fit_transform(y).flatten()

    return train_test_split(x_std, y_std, test_size=test_size)


def main():
    x_train_std, x_test_std, y_train_std, y_test_std = \
        make_data()
    lr = LinearRegressionGD()
    lr.fit(x_train_std, y_train_std)

    return 0


if __name__ == '__main__':
    main()
