import struct
import numpy as np
import pandas as pd

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

X = np.concatenate([X_train, X_valid, X_test], axis=0)
N = X.shape[0] # int32
D = X.shape[1] # int32
theta = 0.5 # double
perplexity = 30.0 # double
no_dims = 3 # int32

with open('data.dat', 'wb') as f:
    f.write(struct.pack('@i', N))
    f.write(struct.pack('@i', D))
    f.write(struct.pack('@d', theta))
    f.write(struct.pack('@d', perplexity))
    f.write(struct.pack('@i', no_dims))
    f.write(X.tobytes())
