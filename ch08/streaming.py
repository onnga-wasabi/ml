import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


def tokenizer(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    stop = stopwords.words('english')
    token = [word for word in text.split() if word not in stop]
    return token


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(getter, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(getter)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


def main():
    vect = HashingVectorizer(n_features=2**21, tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', max_iter=1)
    getter = stream_docs('./train_random_data.csv')
    for _ in range(45):
        x_train, y_train = get_minibatch(getter, 500)
        if not x_train:
            break
        x_train = vect.transform(x_train)
        # onlineでデータを食わせるので、予めクラス数を指定する？
        clf.partial_fit(x_train, y_train, classes=np.array([1, 0]))
    getter = stream_docs('./test_random_data.csv')
    x_test, y_test = get_minibatch(getter, size=2500)
    x_test = vect.transform(x_test)
    print('accuracy:', clf.score(x_test, y_test))
    return 0


if __name__ == '__main__':
    main()
