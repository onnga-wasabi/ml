import numpy as np
import matplotlib.pyplot as plt

# generate data
x1 = np.random.rand(50) * 0.5
x2 = np.random.rand(50) * 0.5 + 0.5

y1 = np.random.rand(50)
y2 = np.random.rand(50)

fig = plt.figure()

ax = fig.add_subplot(2, 2, 4)  # 2x2の4こめ

ax.scatter(x1, y1, c='red', label='group1')
ax.scatter(x2, y2, c='blue', label='group2')

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.legend(loc='upper left')

fig.show()
plt.show()
