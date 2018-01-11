from load_data import load_IMDb
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def main():
    count = CountVectorizer()
    docs = load_IMDb()[0]
    bag = count.fit_transform(docs)
    tfidf = TfidfTransformer()
    print(tfidf.fit_transform(bag).toarray())
    return 0



if __name__ == '__main__':
    main()
