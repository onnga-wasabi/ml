from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key,
                                  value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        predicted_labels = np.array([clf.predict(X)
                                     for clf in self.classifiers_]).T
        maj_vote = np.array([np.argmax(np.bincount(
            label, weights=self.weights)) for label in predicted_labels]).T

        return maj_vote

    def predict_proba(self, X):
        probas = np.array([clf.predict_proba(X)
                           for clf in self.classifiers_])
        # np.average()は重み付き平均を求められる
        avg_probas = np.average(probas, axis=0, weights=self.weights)
        return avg_probas
