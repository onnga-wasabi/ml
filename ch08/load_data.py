import os
import sys
import pandas as pd
import numpy as np


def download():
    '''the func that downloads IMDb with wget, disable on mac'''

    url = 'http://ai.stanford.edu/~amaas/data/sentiment/'
    data = 'aclImdb_v1.tar.gz'
    os.system('wget ' + url + data)
    os.system('tar -xzvf ' + data)
    os.system('rm ' + data)
    return 0


def make_csv():
    labels = {'pos': 1, 'neg': 0}
    for kind in ['train', 'test']:
        df = pd.DataFrame(columns=['review', 'sentiment'])
        path = './aclImdb/'
        path = os.path.join(path, kind)
        for key in labels.keys():
            dir_name = os.path.join(path, key)
            for name in os.listdir(dir_name):
                with open(os.path.join(dir_name, name), 'r', encoding='utf-8') as f:
                    content = f.read()
                series = pd.Series([content, labels[key]], index=df.columns)
                df = df.append(series, ignore_index=True)
        df.to_csv(kind + '_data.csv', index=False)
    return 0


def load_IMDb(kind='train'):
    if 'aclImdb' not in os.listdir('./'):
        download()
    if 'train_data.csv' not in os.listdir('./'):
        make_csv()
    df = pd.read_csv(kind + '_data.csv',  header=None)
    return np.array(df.loc[1:, 0]), np.array(df.loc[1:, 1])


if __name__ == '__main__':
    kind = 'train'
    text, digits = load_IMDb(kind=kind)
    print(kind, "data's length:", text.shape)
