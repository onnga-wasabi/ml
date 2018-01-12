from load_data import load_IMDb
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import re


def preprocessor(doc):
    text = re.sub('<.*?>', '', doc)
    text = re.sub('[\W]+', ' ', text.lower())
    # print(text)
    return text


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in tokenizer(text)]


def remove_stopwords(token):
    '''
    一つの文章に含まれるtokenを渡さないといけまめん
    '''
    stop = stopwords.words('english')
    return [word for word in token if word not in stop]


def main():
    docs = load_IMDb()[0]
    docs = docs[:2]
    docs = np.array([preprocessor(doc) for doc in docs])
    tokens = np.array([tokenizer(doc) for doc in docs])
    #print(tokens)
    #print(np.array([remove_stopwords(token) for token in tokens]))

    return 0


if __name__ == '__main__':
    main()
