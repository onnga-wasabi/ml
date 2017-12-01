import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_wine(test_size=0.3, random_state=0):
    if 'data/wine.csv' in os.getcwd():
        df_wine = pd.read_csv('wine.csv')
    else:
        df_wine = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
        df_wine.to_csv('wine.csv', index=False)

    x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0)

    return x_train, x_test, y_train, y_test


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
    return x.values, y.values


def load_label():
    return ['Class label', 'Alocohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
