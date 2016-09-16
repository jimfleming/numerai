from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd

from tpot import TPOTClassifier

def main():
    df_train = pd.read_csv('data/train_data.csv')
    df_valid = pd.read_csv('data/valid_data.csv')

    feature_cols = list(df_train.columns[:-1])
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values

    tsne_data = np.load('data/tsne_2d_5p.npz')
    tsne_train = tsne_data['X_train']
    tsne_valid = tsne_data['X_valid']

    # concat features
    X_train_concat = np.concatenate([X_train, tsne_train], axis=1)
    X_valid_concat = np.concatenate([X_valid, tsne_valid], axis=1)

    tpot = TPOTClassifier(
        max_time_mins=60 * 24,
        population_size=100,
        scoring='log_loss',
        num_cv_folds=3,
        verbosity=2,
        random_state=67)
    tpot.fit(X_train_concat, y_train)
    print(tpot.score(X_valid_concat, y_valid))
    tpot.export('tpot_pipeline.py')

if __name__ == '__main__':
    main()
