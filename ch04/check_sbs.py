import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from SBS import SBS
import use_colormap
import matplotlib.pyplot as plt
import os


def main():
    if 'wine.csv' in os.getcwd():
        df_wine = pd.read_csv('wine.csv')
    else:
        df_wine = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
        df_wine.to_csv('wine.csv', index=False)

    df_wine.columns = ['Class label', 'Alocohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0)

    # 正規化
    mms = MinMaxScaler()
    x_train_norm = mms.fit_transform(x_train)
    x_test_norm = mms.transform(x_test)

    # 標準化
    stdsc = StandardScaler()
    x_train_std = stdsc.fit_transform(x_train)
    x_test_std = stdsc.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x_train_std,y_train)
    print(knn.score(x_train_std,y_train))
    print(knn.score(x_test_std,y_test))

    sbs = SBS(knn, k_features=1)
    sbs.fit(x_train_std, y_train)
    knn.fit(x_train_std[:,sbs.subsets_[8]],y_train)
    print()
    print(knn.score(x_train_std[:,sbs.subsets_[8]],y_train))
    print(knn.score(x_test_std[:,sbs.subsets_[8]],y_test))


if __name__ == '__main__':
    main()
