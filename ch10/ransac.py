from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.pardir)
from utils.housing import load_housing


def show_ransacline(ransac, x, y, mask):
    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)

    plt.scatter(x[mask['inmask']], y[mask['inmask']],
                marker='o', color='red', label='inliner')
    plt.scatter(x[mask['outmask']], y[mask['outmask']],
                marker='x', color='blue', label='outliner')
    ha = np.arange(x.min(), x.max(), 0.01).reshape(-1, 1)
    y_pred = ransac.predict(ha)
    plt.plot(ha, y_pred, color='orange', label='regression line')
    plt.ylabel('value in $1000\'s')
    plt.xlabel('number of rooms')
    plt.legend()
    plt.pause(1)

    return 0


def main():
    # create dataset
    # 標準化はsklearn->LinearRegeressionの場合勝手にやってくれる
    df = load_housing('../utils/')
    x = df[['RM']].values
    y = df['MEDV'].values

    # create ransac object
    lr = LinearRegression()
    ransac = RANSACRegressor(base_estimator=lr,
                             loss='absolute_loss',
                             residual_threshold=5,
                             max_trials=100,
                             min_samples=50)
    ransac.fit(x, y)
    mask = {}
    mask['inmask'] = ransac.inlier_mask_
    mask['outmask'] = np.logical_not(mask['inmask'])
    show_ransacline(ransac, x, y, mask)

    return 0


if __name__ == '__main__':
    main()
