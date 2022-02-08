from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def get_clf_vectorizers(df):
    count_vect = CountVectorizer(stop_words='english')
    x_train_counts = count_vect.fit_transform(df['comment'])

    # create a tf-idf representation
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # create a naive bayes model
    clf = MultinomialNB().fit(x_train_tfidf, df['label'])

    return clf, count_vect, tfidf_transformer


def predict_from_clf(comments, clf, count_vect, tfidf_transformer):
    x_test_counts = count_vect.transform(comments)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    return clf.predict(x_test_tfidf)


def get_accuracy(pred, labels):
    return sum(pred == labels) / len(pred)
