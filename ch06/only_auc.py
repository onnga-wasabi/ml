from cancer import load_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


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
    pipe_lr.fit(x_train2, y_train)
    y_pred = pipe_lr.predict(x_test[:, [4, 14]])
    print('roc auc score:', roc_auc_score(y_test, y_pred))
    print('accuracy score:', accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
