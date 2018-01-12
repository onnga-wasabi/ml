from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from cleansing import *
from load_data import load_IMDb
import nltk


def main():
    x_train = load_clean_data()
    y_train = load_IMDb()[1]
    x_test = load_clean_data(kind='test')
    y_test = load_IMDb(kind='test')[1]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    stop = nltk.corpus.stopwords.words('english')
    tfidf = TfidfVectorizer(lowercase=False)
    tfidf_lr = Pipeline([('vecs', tfidf), ('clf', LogisticRegression())])
    param_grid = [{'vecs__ngram_range': [(1, 1)],
                   'vecs__stop_words': [stop, None],
                   'vecs__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10, 100]},
                  {'vecs__ngram_range': [(1, 1)],
                   'vecs__stop_words': [stop, None],
                   'vecs__tokenizer': [tokenizer, tokenizer_porter],
                   'vecs__idf':[False],
                   'vecs__norm':[False],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10, 100]}]
    gs_tfidf_lr = GridSearchCV(estimator=tfidf_lr,
                               param_grid=param_grid,
                               scoring='accuracy',
                               n_jobs=-1,
                               cv=5)
    gs_tfidf_lr.fit(x_train, y_train)
    print(gs_tfidf_lr.best_params_)
    print(gs_tfidf_lr.best_score_)
    clf = gs_tfidf_lr.best_estimator_
    print(clf.score(x_test, y_test))

    return 0


if __name__ == '__main__':
    main()
