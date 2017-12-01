import pandas as pd
import os


def load_housing():
    if 'data/housing.csv' in os.listdir('./'):
        df = pd.read_csv('data/housing.csv')
    else:
        df = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
        df.to_csv('data/housing.csv', index=False)
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                  'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATO', 'B', 'LSTAT', 'MEDV']


if __name__ == '__main__':
    load_housing()
