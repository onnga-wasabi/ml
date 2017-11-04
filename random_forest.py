from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

from use_colormap import plot_decision_regions
import numpy as np
import pandas as pd

iris = datasets.load_iris()
y = iris.target
x = iris.data[:, [2, 3]]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)
# 正規化を行わないのは閾値が変わるだけで，２分岐で識別して行く際には分割には影響しないため

#n_estimators->決定木の数
#n_jobs->並列処理をやらせるときのコア数
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1,n_jobs=2)
forest.fit(x_train, y_train)

plot_decision_regions(x_test, y_test, classifier=forest)

