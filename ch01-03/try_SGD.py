from SGD import AdalineSGD
from use_colormap import plot_decision_regions

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# generate datasets
df = pd.read_csv('iris.csv', header=None)
# print(df.tail())

y = df.iloc[:100, 4].values
y = np.where(y == 'Iris-versicolor', 1, -1)
x = df.iloc[:100, [0, 2]].values

# generate AdalineSGD
ppn = AdalineSGD(eta=0.01, n_iter=15)
ppn.fit(x, y)
y_pred = ppn.predict(x)

from sklearn.metrics import accuracy_score
print('Accuracy_metrics: %.2f' % accuracy_score(y, y_pred))

fig = plt.figure()
plane = fig.add_subplot(1, 1, 1)
plane.plot(range(1, len(ppn.cost_) + 1), ppn.cost_)

fig.show()
plt.show()

ppn.accuracy(x, y)
plot_decision_regions(x, y, classifier=ppn)
