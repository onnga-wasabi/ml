import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[:100, 4].values
y = np.where(y == "Iris-versicolor", 1, -1)
x = df.iloc[:100, [0, 2]].values

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.scatter(x[:50, 0], x[:50, 1], c='red', label='Iris-setosa', marker='o')
ax.scatter(x[50:, 0], x[50:, 1], c='blue', label='Iris-versicolor', marker='x')

ax.set_xlabel('sepal length[cm]')
ax.set_ylabel('petal length[cm]')

ax.legend(loc='upper left')

fig.show()
plt.show()
