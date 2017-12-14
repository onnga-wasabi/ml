from neuralnet import NeuralNetMLP
from sklearn.model_selection import train_test_split
from load import load_mnist

import matplotlib.pyplot as plt
import numpy as np


def main():
    images, labels, rows, cols = load_mnist('../')
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2)
    nn = NeuralNetMLP(n_output=10,
                      n_features=x_train.shape[1],
                      n_hidden=50,
                      epochs=10,
                      minibatches=10)
    nn.fit(x_train, y_train)
    y_train_pred = nn.predict(x_train)
    fig, ax = plt.subplots(4, 4)
    fig.canvas.manager.window.attributes('-topmost', 1)
    ax = ax.flatten()
    for i in range(16):
        ax[i].imshow(x_train[i].reshape(rows, cols),
                     cmap='gray', interpolation='bilinear')
        ax[i].set_title(str(y_train[i]) + ' => ' + str(y_train_pred[i]))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == '__main__':
    main()
