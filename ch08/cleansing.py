from load_data import load_IMDb
import numpy as np
import re


def preprocessor(doc):
    text = re.sub('<.*?>', '', doc)
    text = re.sub('[\W]+', ' ', text.lower())
    # print(text)
    return text


def main():
    docs = load_IMDb()[0]
    #docs = docs[:5]
    docs = np.array([preprocessor(doc) for doc in docs])
    print(docs.shape)

    return 0


if __name__ == '__main__':
    main()
