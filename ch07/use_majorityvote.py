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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


def make_data(test_size):
    # roc_aucを使って評価したいので２値分類の問題を解くため、クラスを２つにしてlabelencoderで0,1に変換しておく
    df = load_iris()
    le = LabelEncoder()
    y = df.target[50:]
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test =\
        train_test_split(df.data[50:, [1, 2]], y, test_size=test_size)

    return x_train, x_test, y_train, y_test


def show_roc(x_train, x_test, y_train, y_test, clfs, labels):
    fig = plt.figure()
    linestyles = ['--', ':', '-', '-.']
    for clf, label, line in zip(clfs, labels, linestyles):
        y_pred = clf.fit(x_train, y_train).predict_proba(x_test)
        fpr, tpr, _treshold = roc_curve(y_true=y_test, y_score=y_pred[:, 1])
        roc_auc = auc(x=fpr, y=tpr).round(3)
        plt.plot(fpr, tpr,
                 label=label + ' (auc = ' + str(roc_auc) + ')',
                 linestyle=line)
    fig.canvas.manager.window.attributes('-topmost', 1)
    plt.xlabel('fause positive rate')
    plt.ylabel('true positive rate')
    plt.legend()
    plt.pause(3)

    return 0


def show_decisionline(x_train, x_test, y_train, y_test, clfs, labels):
    x_min = x_test[:, 0].min() - 1
    x_max = x_test[:, 0].max() + 1
    y_min = x_test[:, 1].min() - 1
    y_max = x_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for idx, clf, title in zip(itertools.product([0, 1], [0, 1]), clfs, labels):
        clf.fit(x_train, y_train)
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
        # contourf()でxxに対応して等高線をかくため
        Z = Z.reshape(xx.shape)
        ax[idx].scatter(x_test[y_test == 0, 0],
                        x_test[y_test == 0, 1], marker='^', color='blue')
        ax[idx].scatter(x_test[y_test == 1, 0],
                        x_test[y_test == 1, 1], marker='o', color='red')
        ax[idx].contourf(xx, yy, Z, alpha=0.1)
        ax[idx].set_title(title)
    # 軸を表示したい場合は全てに対して行うか、plt.text(x,y,s,fontsize)で調整するか
    plt.show()

    return 0


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
    class_labels = ['LogisticRegression',
                    'DecisionTree',
                    'KNeighbors']
    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    class_labels += ['MajorityVote']
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    for clf, class_label in zip(all_clf, class_labels):
        score = cross_val_score(estimator=clf,
                                X=x_train,
                                y=y_train,
                                cv=10,
                                scoring='roc_auc')
        print(class_label, np.mean(score).round(4))
    #show_roc(x_train, x_test, y_train, y_test, all_clf, class_labels)
    # print(mv_clf.get_params(deep=False))
    #show_decisionline(x_train, x_test, y_train, y_test, all_clf, class_labels)
    params = {'decisiontreeclassifier__max_depth': [1, 2, 3],
              'pipeline-1__clf__C': [0.001, 0.1, 100]}
    gs = GridSearchCV(estimator=mv_clf,
                      scoring='roc_auc',
                      n_jobs=-1,
                      param_grid=params)
    gs.fit(x_train, y_train)
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score'].round(3)):
        print(score, params)

    return 0


if __name__ == '__main__':
    main()
