from scipy.misc import comb
import math
import numpy as np
import matplotlib.pyplot as plt


def ensemble_error(n_classifier, error):
    ks = math.ceil(n_classifier / 2)
    probs = [comb(n_classifier, k) * error**k * (1 - error) **
             (n_classifier - k)for k in range(ks, n_classifier + 1)]
    return sum(probs)


def show_errors():
    errors = np.linspace(0, 1, 100)
    fig = plt.figure()
    plt.plot(errors, [ensemble_error(n_classifier=11, error=e)
                      for e in errors], label='ensemble errors')
    plt.plot(errors, errors, label='base errors')
    fig.canvas.manager.window.attributes('-topmost', 1)
    plt.xlabel('base errors')
    plt.ylabel('base/ensemble errors')
    plt.legend()
    plt.grid()
    plt.pause(2)


def main():
    show_errors()
    return 0


if __name__ == '__main__':
    main()
