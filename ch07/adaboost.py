from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
sys.path.append(os.pardir)
from utils import wine


def load_data(test_size=0.4):
    if 'wine.csv' in os.listdir('./'):
        df = pd.read_csv('wine.csv')
    else:
        df = pd.read_csv(
            'https://archive.ics.edu/ml/machine-learning-datasets/wine/wine.data', header=None)
        df.to_csv('wine.csv', index=False)
    df.columns = wine.load_label()
    df = df[df['Class label'] != 1]
    y = df['Class label'].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    x = df[['Alcohol', 'Hue']].values

    return train_test_split(x, y, test_size=test_size)


def show_decisionline(tree, ada, x_test, y_test):
    x_min = x_test[:, 0].min() - 1
    x_max = x_test[:, 0].max() + 1
    y_min = x_test[:, 1].min() - 1
    y_max = x_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    fig.canvas.manager.window.attributes('-topmost', 1)
    titles = ['tree', 'ada']
    for i, clf in enumerate([tree, ada]):
        clf.fit(x_test, y_test)
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)
        ax[i].contourf(xx, yy, Z, alpha=0.2)
        ax[i].scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1])
        ax[i].scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1])
        ax[i].set_title(titles[i])
    plt.pause(0.5)


def main():
    x_train, x_test, y_train, y_test = load_data()
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(x_train, y_train)
    print('Decision Tree', tree.score(x_test, y_test).round(3))

    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=500,
                             learning_rate=0.1)
    ada.fit(x_train, y_train)
    print('AdaBoost     ', ada.score(x_test, y_test).round(3))
    show_decisionline(tree, ada, x_test, y_test)
    print(tree)

    return 0


if __name__ == '__main__':
    main()
