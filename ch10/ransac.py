from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

import sys
import os
sys.path.append(os.pardir)
from utils.housing import load_housing


def main():
    df = load_housing('../utils/')
    x = df[['RM']].values
    y = df['MEDV'].values
    lr = LinearRegression()
    ransac = RANSACRegressor(base_estimator=lr,
                             loss='absolute_loss',
                             residual_threshold=5,
                             max_trials=100,
                             min_samples=50)
    print(x.shape, y.shape)
    return 0


if __name__ == '__main__':
    main()
