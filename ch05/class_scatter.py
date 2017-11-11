from wine import *
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt


def main():
    x_train, x_test, y_train, y_test = load_wine()
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)

    x_mean = np.array([np.mean(x_train_std[y_train == label], axis=0)
                       for label in np.unique(y_train)])

    d = x_train_std.shape[1]
    S_W = np.zeros((d, d))
    for label, mv in zip(np.unique(y_train), x_mean):
        for data in x_train_std[y_train == label]:
            # 縦ベクトルへの変換を行わないとクラス内変動行列を作るのに苦労する
            # そのままnp.dotを使うと内積が計算される
            scat = np.vstack(data - mv)
            S_W += scat.dot(scat.T)
    print(S_W)
    #せっかく計算したけどクラスラベルが一様に分布していない、クラスラベルの個数が異なるので、クラスごとスケーリングをおこなうことを考えると、この変動行列は結局のところ共分散行列となることがわかる。LDA.pyにてnp.covを使って共分散行列を作成し、続ける。


if __name__ == '__main__':
    main()
