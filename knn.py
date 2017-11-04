from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from use_colormap import plot_decision_regions
import numpy as np
import pandas as pd

iris = datasets.load_iris()
y = iris.target
x = iris.data[:, [2, 3]]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)
# 正規化を行わないのは閾値が変わるだけで，２分岐で識別して行く際には分割には影響しないため

#n_neighbors->近傍の点の数
#p->距離(2:ユーグリッド、1:マンハッタン)
#metric->先のpの値とともに用いる。minkowskiはユーグリッド距離とマンハッタン距離一般化したもの
knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn.fit(x_train, y_train)

plot_decision_regions(x_test, y_test, classifier=knn)

