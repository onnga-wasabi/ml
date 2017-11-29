from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def main():
    x_train, x_test, y_train, y_test = make_data(test_size=0.2)
    clf1 = LogisticRgression(penalty='l2',
                             C=0.001)
    clf2 = DecisionTreeClassifier(max_depth=1,
                                  criterion='entropy')
    clf3 = KNeighborsClassifier(n_neighbors=5)
    pipe1 = Pipeline([('sc', StandardScaler()),
                      ('clf', clf1)])
    pipe3 = Pipeline([('sc', StandardScaler()),
                      ('clf', clf3)])
    return 0


def make_data(test_size):
    df = load_iris()
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    make_data(1)
