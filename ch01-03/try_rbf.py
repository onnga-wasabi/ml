from sklearn.svm import SVC

from use_colormap import plot_decision_regions
import numpy as np

x = np.random.randn(200, 2)
y = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)
y = np.where(y, 1, -1)

svm = SVC(kernel='rbf', gamma=0.1, C=10, random_state=0)
svm.fit(x, y)
plot_decision_regions(x, y, classifier=svm)
