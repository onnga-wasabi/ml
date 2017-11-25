import pandas as pd
import os


def load_cancer():
    if 'cancer.csv' in os.listdir():
        df = pd.read_csv('cancer.csv')
    else:
        df = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
        df.to_csv('cancer.csv')

    # df.info()
    y = df.iloc[:, 2]
    x = df.iloc[:, 3:]
    return x, y


if __name__ == '__main__':
    load_cancer()
