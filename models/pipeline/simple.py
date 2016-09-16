from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time
import random
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from transformers import ItemSelector

def main():
    # load data
    df_train = pd.read_csv('data/train_data.csv')
    df_valid = pd.read_csv('data/valid_data.csv')
    df_test = pd.read_csv('data/test_data.csv')

    feature_cols = list(df_train.columns[:-1])
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values

    X_test = df_test[feature_cols].values

    classifier = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2)),
        ('lr', LogisticRegression(penalty='l2', C=1e-2, n_jobs=-1, verbose=2)),
    ])

    print('Fitting...')
    start_time = time.time()
    classifier.fit(X_train, y_train)
    print('Fit: {}s'.format(time.time() - start_time))

    p_valid = classifier.predict_proba(X_valid)
    loss = log_loss(y_valid, p_valid)
    print('Loss: {}'.format(loss))

    p_test = classifier.predict_proba(X_test)
    df_pred = pd.DataFrame({
        't_id': df_test['t_id'],
        'probability': p_test[:,1]
    })
    csv_path = 'predictions/predictions_{}_{}.csv'.format(int(time.time()), loss)
    df_pred.to_csv(csv_path, columns=('t_id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
