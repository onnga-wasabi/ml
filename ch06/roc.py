from cancer import load_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from scipy import interp
import numpy as np

import matplotlib.pyplot as plt


def main():
    x, y = load_cancer()
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3)
    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=2)),
                        ('clf', LogisticRegression(
                            penalty='l2',
                            C=100))])

    # rocの効果をわかり安くするためデータの要素を減らして実験
    x_train2 = x_train[:, [4, 14]]

    fig = plt.figure()
    fig.canvas.manager.window.attributes('-topmost', 1)

    tprs = 0
    mean_fprs = np.linspace(0, 1, 100)

    cv = StratifiedKFold(n_splits=3)
    for i, (train, test) in enumerate(cv.split(x_train, y_train)):
        # predict_probaは各ラベルである確率を返す
        probas = pipe_lr.fit(
            x_train2[train], y_train[train]).predict_proba(x_train2[test])

        # roc_curveは正解ラベルとしてした陽性ラベルの可能性を指定するとfalse positive,true positive,thresholdの順で値を返す.thresholdはその時の境界の値たち？
        fpr, tpr, thresholds = roc_curve(
            y_train[test], probas[:, 1], pos_label=1)

        # 毎回得られるfprとtprはランダムなのでサンプル点を揃えるため0-100に線形射影してしまってあとで帳尻を合わせる
        tprs += interp(mean_fprs, fpr, tpr)
        # 曲線下の面積を計算area under the curve
        roc_auc = auc(fpr, tpr).round(3)
        plt.plot(fpr, tpr, lw=1, label=str(i + 1) +
                 'fold\'s roc auc:' + str(roc_auc))

    # 平均のroc,aucを表示
    mean_auc = auc(mean_fprs, (tprs / cv.n_splits)).round(3)
    plt.plot(mean_fprs,
             (tprs / cv.n_splits),
             label='mean roc auc:' + str(mean_auc),
             linestyle='--',
             lw=2)

    # 当て推量(適当に分類したとき)の曲線->対角線になる
    plt.plot([0, 1],
             [0, 1],
             label='random guessing',
             linestyle='--')

    # perfectなperformanceのとき
    plt.plot([0, 0, 1],
             [0, 1, 1],
             label='perfect performance',
             linestyle=':')


    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.ylabel('true positive rate')
    plt.xlabel('false positive rate')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
