from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

svm=SVC(kernel='linear',C=1,random_state=0)
svm.fit(x_train_std,y_train)
plot_decision_regions(x_test_std, y_test, classifier=svm)
