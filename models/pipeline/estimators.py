from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA

def _get_portion(X, y, portion):
    num_samples = len(X)
    num_portion = int(num_samples * portion) \
        if isinstance(portion, float) \
        else portion
    idx = np.random.choice(num_samples, num_portion)
    X_ = X[idx]
    y_ = y[idx] if y is not None else y
    return X_, y_

class PortionKernelPCA(KernelPCA):
    def __init__(self, n_components=None, kernel='linear', gamma=None, \
                 degree=3, coef0=1, kernel_params=None, alpha=1.0, \
                 fit_inverse_transform=False, eigen_solver='auto', \
                 tol=0, max_iter=None, remove_zero_eig=False, portion=0.1):
        super(PortionKernelPCA, self).__init__(n_components, kernel, \
              gamma, degree, coef0, \
              kernel_params, alpha, \
              fit_inverse_transform, \
              eigen_solver, tol, \
              max_iter, remove_zero_eig)
        self.portion = portion

    def fit(self, X, y=None):
        X_, y_ = _get_portion(X, y, self.portion)
        super(PortionKernelPCA, self).fit(X_, y_)
        return self

    def fit_transform(self, X, y=None):
        X_, y_ = _get_portion(X, y, self.portion)
        super(PortionKernelPCA, self).fit(X_, y_)
        return self.transform(X)

class PortionIsomap(Isomap):
    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto', \
                 tol=0, max_iter=None, path_method='auto', \
                 neighbors_algorithm='auto', portion=0.1):
        super(PortionIsomap, self).__init__(n_neighbors, n_components, \
              eigen_solver, tol, max_iter, \
              path_method, neighbors_algorithm)
        self.portion = portion

    def fit(self, X, y=None):
        X_, y_ = _get_portion(X, y, self.portion)
        super(PortionIsomap, self).fit(X_, y_)
        return self

    def fit_transform(self, X, y=None):
        X_, y_ = _get_portion(X, y, self.portion)
        super(PortionIsomap, self).fit(X_, y_)
        return self.transform(X)
