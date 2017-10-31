from Perceptron import Perceptron
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

# generate Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)

# generate figure
plot_decision_regions(x, y, classifier=ppn)
