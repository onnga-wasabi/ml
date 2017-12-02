import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
sys.path.append(os.pardir)
from utils.housing import load_housing


def main():
    df = load_housing(path='../utils/')
    sns.set(style='whitegrid', context='notebook')
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    sns.pairplot(df[cols])
    plt.show()
    return 0


if __name__ == '__main__':
    main()
