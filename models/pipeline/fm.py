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

from scipy.sparse import csc_matrix

from fastFM.als import FMClassification

from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.feature_selection import SelectKBest

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
    y_train[y_train == 0] = -1

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values
    y_valid[y_valid == 0] = -1

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

    # build pipeline
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

    fm = FMClassification(n_iter=300, rank=8, l2_reg_w=1e-2, l2_reg_V=1e-2)

    print('Fitting...')
    start_time = time.time()
    fm.fit(csc_matrix(pipeline.fit_transform(X_train_concat, y_train)), y_train)
    print('Fit: {}s'.format(time.time() - start_time))

    p_valid = fm.predict_proba(csc_matrix(pipeline.transform(X_valid_concat)))
    loss = log_loss(y_valid, p_valid)
    print('Loss: {}'.format(loss))

    p_test = fm.predict_proba(csc_matrix(pipeline.transform(X_test_concat)))
    df_pred = pd.DataFrame({
        't_id': df_test['t_id'],
        'probability': p_test
    })
    csv_path = 'predictions/predictions_{}_{}.csv'.format(int(time.time()), loss)
    df_pred.to_csv(csv_path, columns=('t_id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
