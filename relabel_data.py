import os

import pandas as pd
from sklearn.utils import shuffle

from utils.models import predict_from_clf, get_clf_vectorizers, get_accuracy

if __name__ == '__main__':
    # %% Loads Data
    folder = 'Clean_Data'
    files = ['conservatives.csv', 'democrats.csv']
    labels = [1, -1]

    test_p = 0.15

    dfs_train, dfs_test = [], []
    for file, label in zip(files, labels):
        df = pd.read_csv(os.path.join(folder, file))
        df['words'] = df['comment'].str.split().str.len()
        df['label'] = label
        dfs_train.append(df.iloc[:-int(df.shape[0] * test_p)])
        dfs_test.append(df.iloc[-int(df.shape[0] * test_p):])

    ds_train = pd.concat(dfs_train, axis=0)
    ds_test = pd.concat(dfs_test, axis=0)
    ds_train['comment'] = ds_train['comment'].str.lower()
    ds_test['comment'] = ds_test['comment'].str.lower()

    ds_test = shuffle(ds_test)
    ds_train = shuffle(ds_train)

    # %% Training
    clf, count_vect, tfidf_transformer = get_clf_vectorizers(ds_train)

    # %% Predict
    test_prediction = predict_from_clf(
        ds_test['comment'], clf, count_vect, tfidf_transformer)
    train_prediction = predict_from_clf(
        ds_train['comment'], clf, count_vect, tfidf_transformer)

    # %% Accuracies
    train_acc = get_accuracy(train_prediction, ds_train['label'])
    print('train accuracy: {}'.format(train_acc))

    test_acc = get_accuracy(test_prediction, ds_test['label'])
    print('test accuracy: {}'.format(test_acc))

    # %%
    all_data = pd.concat([ds_train, ds_test], axis=0)

    X_counts = count_vect.transform(all_data['comment'])
    X_tfidf = tfidf_transformer.transform(X_counts)

    prediction = clf.predict(X_tfidf)
    prediction_proba = clf.predict_proba(X_tfidf)

    df = pd.DataFrame()
    df['pred'] = prediction
    df['proba0'] = prediction_proba[:, 0]
    df['proba1'] = prediction_proba[:, 1]

    df['new_label'] = df['pred']
    df.loc[(df['proba0'] < 0.6) & (df['proba0'] > 0.4), 'new_label'] = 0

    all_data.reset_index(inplace=True)
    new_dataset = pd.concat([df, all_data], axis=1)
    new_dataset.to_csv(os.path.join('Clean_Data', 'all_data_relabeled.csv'))
