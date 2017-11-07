from wine import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


def main():
    x_train, x_test, y_train, y_test = load_wine()
    forest = RandomForestClassifier(
        n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(x_train, y_train)
    features = forest.feature_importances_

    indices = np.argsort(forest.feature_importances_)[::-1]
    labels = load_label()[1:]
    sorted_labels = []
    print('importances are')
    for i in range(len(labels)):
        print('%2d) %-*s %f' %
              (i + 1, 30, labels[indices[i]], features[indices[i]]))
        sorted_labels.append(labels[indices[i]])

    fig = plt.figure()
    plt.bar(range(len(labels)), features[indices])
    plt.xticks(range(len(labels)), sorted_labels, rotation=90)
    plt.tight_layout()
    fig.show()
    plt.show()


if __name__ == '__main__':
    main()
