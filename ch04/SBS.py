from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class SBS():
    '''
    逐次後退選択 Sequential backward selection 実行クラス
    '''

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.estimator = estimator
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state)
        dim = x.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets = [self.indices_]
        score = self._calc_score(
            x_train, y_train, x_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score_(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subusets_.append(self.indices_)
            dim = -1
            self.scores_append(socre[best])
        self.k_scores_ = self.scores[-1]
        return self

    def transform(self, x):
        return x[:, self.indeces_]

    def _calc_score_(self, x_train, y_train, x_test, y_test, indices):
        self.estimator.fit(x_train[:, indices], y_train)
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
