from linear import LinearRegressionGD
import sys
import os
sys.path.append(os.pardir)

from utils.wine import load_wine


def main():
    x_train, x_test, y_train, y_test = load_wine()
    lr = LinearRegressionGD()
    lr.fit(x_train, y_train)

    return 0


if __name__ == '__main__':
    main()
