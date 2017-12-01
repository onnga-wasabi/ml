from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from utils import wine


def load_data(test_size=0.3):
    if 'wine.csv' in os.listdir('./'):
        df = pd.read_csv('wine.csv')
    else:
        df = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
        df.to_csv('wine.csv', index=False)
    df.columns = wine.load_label()
    df = df[df['Class label'] != 1]
    y = df['Class label'].values
    x = df[['Alcohol', 'Hue']].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def show_decision_region(tree, bag, x_test, y_test):
    fig, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].plot(x_test[y_test == 0, 0], x_test[y_test == 0, 1])
    return 0


def main():
    x_train, x_test, y_train, y_test = load_data(test_size=0.4)
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(x_train, y_train)
    print('validation score')
    print('DecisionTree', tree.score(x_train, y_train).round(3))
    bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=500,
                            n_jobs=-1)
    bag.fit(x_train, y_train)
    print('Bagging', bag.score(x_train, y_train).round(3))
    print()
    print('test score')
    print('DecisionTree', tree.score(x_test, y_test).round(3))
    print('Bagging', bag.score(x_test, y_test).round(3))
    return 0


if __name__ == '__main__':
    main()
