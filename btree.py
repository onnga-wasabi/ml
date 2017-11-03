import matplotlib.pyplot as plt
import numpy as np


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def error(p):
    return 1 - np.max([p, 1 - p])


def main():
    x = np.arange(0.0, 1.0, 0.01)
    gin = gini(x)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [p * 0.5 if p else None for p in ent]
    err = [error(p) for p in x]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab in zip([gin, ent, sc_ent, err],
                        ['gini', 'entropy', 'scaled entropy', 'error']):
        ax.plot(x, i,label=lab)
    plt.xlim([0,1])
    plt.xlabel('p(i=1):btree')
    fig.show()
    plt.show()


if __name__ == '__main__':
    main()
