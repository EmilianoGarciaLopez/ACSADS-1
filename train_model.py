import os

import joblib
import pandas as pd
from sklearn.utils import shuffle

from utils.models import predict_from_clf, get_clf_vectorizers, get_accuracy

if __name__ == '__main__':
    # %%
    folder = 'Clean_Data'
    # files = ['conservatives.csv', 'democrats.csv']
    # labels = [1, -1]

    test_p = 0.15  # the percentage of the data that will be used for testing

    # dfs_train, dfs_test = [], []
    # for file, label in zip(files, labels):
    #     df = pd.read_csv(os.path.join(folder, file))
    #     df['words'] = df['comment'].str.split().str.len()
    #     df['label'] = label
    #     dfs_train.append(df.iloc[:-int(df.shape[0] * test_p)])
    #     dfs_test.append(df.iloc[-int(df.shape[0] * test_p):])

    # ds_train = pd.concat(dfs_train, axis=0)
    # ds_test = pd.concat(dfs_test, axis=0)
    df = pd.read_csv(os.path.join(folder, 'all_data_relabeled.csv'))
    # df['label'] = df['new_label']

    ds_train = df.iloc[:-int(df.shape[0] * test_p)]
    ds_test = (df.iloc[-int(df.shape[0] * test_p):])

    # ds_train['comment'] = ds_train['comment'].str.lower()
    # ds_test['comment'] = ds_test['comment'].str.lower()

    ds_test = shuffle(ds_test)
    ds_train = shuffle(ds_train)

    # %% Training
    clf, count_vect, tfidf_transformer = get_clf_vectorizers(ds_train)

    # %% Predict
    test_prediction = predict_from_clf(ds_test['comment'], clf, count_vect, tfidf_transformer)
    train_prediction = predict_from_clf(ds_train['comment'], clf, count_vect, tfidf_transformer)

    # %% Accuracies
    train_acc = get_accuracy(train_prediction, ds_train['label'])
    print(f'train accuracy: {train_acc}')

    test_acc = get_accuracy(test_prediction, ds_test['label'])
    print(f'test accuracy: {test_acc}')

    # %%
    model_folder = 'models'
    version = 'retest'
    os.makedirs(model_folder, exist_ok=True)

    joblib.dump([clf, count_vect, tfidf_transformer], os.path.join(model_folder,
                                                                   f'NV_v{version}.z'))
