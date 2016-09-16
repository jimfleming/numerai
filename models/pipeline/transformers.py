from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
