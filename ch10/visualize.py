import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import sys
sys.path.append(os.pardir)
from utils.housing import load_housing


def show_scatter(df):
    sns.set(style='whitegrid', context='notebook')
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    sns.pairplot(df[cols])
    plt.show()

    return 0


def show_correlation_matrix(df):
    sns.set()
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    cm = np.corrcoef(df[cols].values.T)
    sns.heatmap(cm,
                alpha=0.8,
                center=True,
                annot=True,
                xticklabels=cols,
                yticklabels=cols)
    plt.show()

    return 0


def main():
    df = load_housing(path='../utils/')
    show_scatter(df)
    show_correlation_matrix(df)

    return 0


if __name__ == '__main__':
    main()
