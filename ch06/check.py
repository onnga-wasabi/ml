from cancer import load_cancer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main():
    x, y = load_cancer()
    #print(y[0], '\n', x[0])
    #print(np.where(y == 'M'))

    le = LabelEncoder()
    y = le.fit_transform(y)
    #print('[M B]:', le.transform(['M', 'B']))
    #print(np.where(y == 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(
        n_components=2)), ('clr', LogisticRegression())])
    pipe_lr.fit(x_train, y_train)
    print('Test accuracy:', pipe_lr.score(x_test, y_test))


if __name__ == '__main__':
    main()
