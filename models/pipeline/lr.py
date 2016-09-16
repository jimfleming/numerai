from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time
import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd

from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.manifold import Isomap

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

    tsne_data_10p = np.load('data/tsne_2d_10p_poly.npz')
    tsne_data_20p = np.load('data/tsne_2d_20p_poly.npz')
    tsne_data_30p = np.load('data/tsne_2d_30p_poly.npz')
    tsne_data_40p = np.load('data/tsne_2d_40p_poly.npz')
    tsne_data_50p = np.load('data/tsne_2d_50p_poly.npz')

    # concat features
    X_train_concat = {
        'X': X_train,
        'tsne_10p': tsne_data_10p['train'],
        'tsne_20p': tsne_data_20p['train'],
        'tsne_30p': tsne_data_30p['train'],
        'tsne_40p': tsne_data_40p['train'],
        'tsne_50p': tsne_data_50p['train'],
    }
    X_valid_concat = {
        'X': X_valid,
        'tsne_10p': tsne_data_10p['valid'],
        'tsne_20p': tsne_data_20p['valid'],
        'tsne_30p': tsne_data_30p['valid'],
        'tsne_40p': tsne_data_40p['valid'],
        'tsne_50p': tsne_data_50p['valid'],
    }
    X_test_concat = {
        'X': X_test,
        'tsne_10p': tsne_data_10p['test'],
        'tsne_20p': tsne_data_20p['test'],
        'tsne_30p': tsne_data_30p['test'],
        'tsne_40p': tsne_data_40p['test'],
        'tsne_50p': tsne_data_50p['test'],
    }

    # build pipeline
    classifier = Pipeline(steps=[
        ('features', FeatureUnion(transformer_list=[
            ('tsne_10p', ItemSelector('tsne_10p')),
            ('tsne_20p', ItemSelector('tsne_20p')),
            ('tsne_30p', ItemSelector('tsne_30p')),
            ('tsne_40p', ItemSelector('tsne_40p')),
            ('tsne_50p', ItemSelector('tsne_50p')),
            ('X', ItemSelector('X')),
        ])),
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', MinMaxScaler()),
        ('lr', LogisticRegression(penalty='l2', C=1e-2, n_jobs=-1)),
    ])

    print('Fitting...')
    start_time = time.time()
    classifier.fit(X_train_concat, y_train)
    print('Fit: {}s'.format(time.time() - start_time))

    p_valid = classifier.predict_proba(X_valid_concat)
    loss = log_loss(y_valid, p_valid)
    print('Loss: {}'.format(loss))

    p_test = classifier.predict_proba(X_test_concat)
    df_pred = pd.DataFrame({
        't_id': df_test['t_id'],
        'probability': p_test[:,1]
    })
    csv_path = 'predictions/predictions_{}_{}.csv'.format(int(time.time()), loss)
    df_pred.to_csv(csv_path, columns=('t_id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
