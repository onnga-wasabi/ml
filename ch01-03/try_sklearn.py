from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

from use_colormap import plot_decision_regions
import numpy as np
import pandas as pd

iris = datasets.load_iris()
y = iris.target
x = iris.data[:, [2, 3]]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0, shuffle=True)
ppn.fit(x_train_std, y_train)
y_pred = ppn.predict(x_test_std)
print('Accuracy:' + str((len(y_test) - (y_test != y_pred).sum()) / len(y_test)))

from sklearn.metrics import accuracy_score
print('Accuracy_metris: %.2f' % accuracy_score(y_test, y_pred))

ppn.accuracy(x, y)
plot_decision_regions(x_test_std, y_test, classifier=ppn)
