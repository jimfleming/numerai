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

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline, make_union, Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.externals import joblib

from transformers import ItemSelector

from tqdm import trange

def divide_samples_test(X):
    return {
        'L': X,
        'R': shuffle(X),
    }

def divide_samples_train(X, y):
    X1 = X[y == 1]
    X0 = X[y == 0]

    y1 = y[y == 1]
    y0 = y[y == 0]

    # trim by minimum number of samples between sets
    l = min(len(y0), len(y1))

    X_L = np.concatenate([X1[:l], X0[:l]], axis=0)
    X_R = np.concatenate([X0[:l], X1[:l]], axis=0)
    X_both = {
        'L': X_L,
        'R': X_R,
    }

    y_both = np.concatenate([y1[:l], y0[:l]], axis=0)

    return X_both, y_both

def main():
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

    tsne_data_2d_5p = np.load('data/tsne_2d_5p.npz')
    tsne_data_2d_10p = np.load('data/tsne_2d_10p.npz')
    tsne_data_2d_15p = np.load('data/tsne_2d_15p.npz')
    tsne_data_2d_20p = np.load('data/tsne_2d_20p.npz')
    tsne_data_2d_30p = np.load('data/tsne_2d_30p.npz')
    tsne_data_2d_40p = np.load('data/tsne_2d_40p.npz')
    tsne_data_2d_50p = np.load('data/tsne_2d_50p.npz')
    tsne_data_3d_30p = np.load('data/tsne_3d_30p.npz')

    # concat features
    X_train_concat = {
        'X': X_train,
        'tsne_2d_5p': tsne_data_2d_5p['train'],
        'tsne_2d_10p': tsne_data_2d_10p['train'],
        'tsne_2d_15p': tsne_data_2d_15p['train'],
        'tsne_2d_20p': tsne_data_2d_20p['train'],
        'tsne_2d_30p': tsne_data_2d_30p['train'],
        'tsne_2d_40p': tsne_data_2d_40p['train'],
        'tsne_2d_50p': tsne_data_2d_50p['train'],
        'tsne_3d_30p': tsne_data_3d_30p['train'],
    }
    X_valid_concat = {
        'X': X_valid,
        'tsne_2d_5p': tsne_data_2d_5p['valid'],
        'tsne_2d_10p': tsne_data_2d_10p['valid'],
        'tsne_2d_15p': tsne_data_2d_15p['valid'],
        'tsne_2d_20p': tsne_data_2d_20p['valid'],
        'tsne_2d_30p': tsne_data_2d_30p['valid'],
        'tsne_2d_40p': tsne_data_2d_40p['valid'],
        'tsne_2d_50p': tsne_data_2d_50p['valid'],
        'tsne_3d_30p': tsne_data_3d_30p['valid'],
    }
    X_test_concat = {
        'X': X_test,
        'tsne_2d_5p': tsne_data_2d_5p['test'],
        'tsne_2d_10p': tsne_data_2d_10p['test'],
        'tsne_2d_15p': tsne_data_2d_15p['test'],
        'tsne_2d_20p': tsne_data_2d_20p['test'],
        'tsne_2d_30p': tsne_data_2d_30p['test'],
        'tsne_2d_40p': tsne_data_2d_40p['test'],
        'tsne_2d_50p': tsne_data_2d_50p['test'],
        'tsne_3d_30p': tsne_data_3d_30p['test'],
    }

    pipeline = Pipeline(steps=[
        ('features', FeatureUnion(transformer_list=[
            ('X', ItemSelector('X')),
            ('tsne_2d_5p', ItemSelector('tsne_2d_5p')),
            ('tsne_2d_10p', ItemSelector('tsne_2d_10p')),
            ('tsne_2d_15p', ItemSelector('tsne_2d_15p')),
            ('tsne_2d_20p', ItemSelector('tsne_2d_20p')),
            ('tsne_2d_30p', ItemSelector('tsne_2d_30p')),
            ('tsne_2d_40p', ItemSelector('tsne_2d_40p')),
            ('tsne_2d_50p', ItemSelector('tsne_2d_50p')),
            ('tsne_3d_30p', ItemSelector('tsne_3d_30p')),
        ])),
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', MinMaxScaler()),
    ])

    X_train_concat = pipeline.fit_transform(X_train_concat, y_train)
    X_valid_concat = pipeline.transform(X_valid_concat)
    X_test_concat = pipeline.transform(X_test_concat)

    X_valid_both, y_valid_both = divide_samples_train(X_valid_concat, y_valid)

    classifier = make_pipeline(make_union(
        ItemSelector(key='L'),
        ItemSelector(key='R')
    ), LogisticRegression(penalty='l2', C=1e-2, n_jobs=-1, warm_start=True))

    for i in trange(10):
        X_train_both, y_train_both = divide_samples_train(*shuffle(X_train_concat, y_train))

        print('Fitting...')
        start_time = time.time()
        classifier.fit(X_train_both, y_train_both)
        print('Fit: {}s'.format(time.time() - start_time))

        p_valid = classifier.predict_proba(X_valid_both)
        loss = log_loss(y_valid_both, p_valid[:,1])
        auc = roc_auc_score(y_valid_both, p_valid[:,1])
        print('Pairwise Loss: {}, AUC: {}'.format(loss, auc))

    p_valids = []
    for i in trange(100):
        X_valid_both = divide_samples_test(X_valid_concat)
        p_valid = classifier.predict_proba(X_valid_both)
        p_valids.append(p_valid)
    p_valid = np.array(p_valids)
    p_valid = np.mean(p_valid, axis=0)

    loss = log_loss(y_valid, p_valid[:,1])
    auc = roc_auc_score(y_valid, p_valid[:,1])
    print('Validation Loss: {}, AUC: {}'.format(loss, auc))

    p_tests = []
    for i in trange(100):
        X_test_both = divide_samples_test(X_test_concat)
        p_test = classifier.predict_proba(X_test_both)
        p_tests.append(p_test)
    p_test = np.array(p_tests)
    p_test = np.mean(p_test, axis=0)

    df_pred = pd.DataFrame({
        't_id': df_test['t_id'],
        'probability': p_test[:,1]
    })
    csv_path = 'predictions/predictions_{}_{}.csv'.format(int(time.time()), loss)
    df_pred.to_csv(csv_path, columns=('t_id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
