from wine import load_wine
from sklearn.datasets import make_moons
from rbf import rbf_kernel_pca


def main():
    x_train, x_test, y_train, y_test = load_wine()
    x, y = make_moons()
    rbf_kernel_pca(x, 1, 2)


if __name__ == '__main__':
    main()
