from linear import LinearRegressionGD
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.pardir)

from utils.wine import load_wine


def main():
    x_train, x_test, y_train, y_test = load_wine()
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)
    lr = LinearRegressionGD()
    lr.fit(x_train_std, y_train)

    return 0


if __name__ == '__main__':
    main()
