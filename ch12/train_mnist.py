from neuralnet import NeuralNetMLP
from sklearn.model_selection import train_test_split
from load import load_mnist


def main():
    images, labels, rows, cols = load_mnist('../')
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2)
    nn = NeuralNetMLP(n_output=10,
                      n_features=x_train.shape[1],
                      n_hidden=50,
                      epochs=500,
                      minibatches=10)
    nn.fit(x_train, y_train)
    return 0


if __name__ == '__main__':
    main()
