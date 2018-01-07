import os
import sys


def download():
    '''the func that downloads IMDb with wget, disable on mac'''

    url = 'http://ai.stanford.edu/~amaas/data/sentiment/'
    data = 'aclImdb_v1.tar.gz'
    os.system('wget ' + url + data)
    os.system('tar -xzvf ' + data)
    os.system('rm ' + data)
    return 0


if __name__ == '__main__':
    if 'aclImdb' in os.listdir('./'):
        print('dataset already exits in current dir')
        sys.exit()
    download()
