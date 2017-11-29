from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from Classifier import MajorityVoteClassifier
import pandas as pd
import numpy as np


def main():
    x_train, x_test, y_train, y_test = make_data(test_size=0.3)
    clf1 = LogisticRegression(penalty='l2',
                              C=0.001)
    clf2 = DecisionTreeClassifier(max_depth=1,
                                  criterion='entropy')
    clf3 = KNeighborsClassifier(n_neighbors=1)
    pipe1 = Pipeline([('sc', StandardScaler()),
                      ('clf', clf1)])
    pipe3 = Pipeline([('sc', StandardScaler()),
                      ('clf', clf3)])

    # 表示用クラスラベルリスト
    class_label = ['LogisticRegression',
                   'DecisionTree',
                   'KNeighbors']
    '''
    for clf, label in zip([clf1, clf2, clf3], class_label):
        score = cross_val_score(estimator=clf,
                                X=x_train,
                                y=y_train,
                                cv=10,
                                scoring='roc_auc')
        print(label, np.mean(score))
    print()
    '''
    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    class_label += ['MajorityVote']
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    for clf, class_label in zip(all_clf, class_label):
        score = cross_val_score(estimator=clf,
                                X=x_train,
                                y=y_train,
                                cv=10,
                                scoring='roc_auc')
        print(class_label, np.mean(score).round(4))
    return 0


def make_data(test_size):
    # roc_aucを使って評価したいので２値分類の問題を解くため、クラスを２つにしてlabelencoderで0,1に変換しておく
    df = load_iris()
    le = LabelEncoder()
    y = df.target[50:]
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test =\
        train_test_split(df.data[50:, [1, 2]], y, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    main()
