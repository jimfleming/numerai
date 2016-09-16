from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

def main():
    params = {
        # 'featureunion__polynomialfeatures__degree': range(2, 4),
        # 'featureunion__portionkernelpca__n_components': range(2, 102, 2),
        # 'featureunion__portionkernelpca__degree': range(2, 4),
        # 'featureunion__portionkernelpca__kernel': ['cosine', 'rbf'],
        # 'featureunion__portionisomap__n_neighbors': range(1, 11),
        # 'featureunion__portionisomap__n_components': range(2, 102, 2),
        'pca__n_components': range(2, 202, 2),
        'pca__whiten': [True, False],
        'logisticregression__C': [1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
        'logisticregression__penalty': ['l1', 'l2']
    }

    pipeline = build_pipeline(portion=0.1)

    X_search = np.concatenate([
            np.concatenate([X_train, X_train_tsne], axis=1),
                np.concatenate([X_valid, X_valid_tsne], axis=1),
                ], axis=0)
    y_search = np.concatenate([y_train, y_valid], axis=0)

    train_indices = range(0, len(X_train))
    valid_indices = range(len(X_train), len(X_train)+len(X_valid))
    assert(len(train_indices) == len(X_train))
    assert(len(valid_indices) == len(X_valid))

    cv = [(train_indices, valid_indices)]

    search = RandomizedSearchCV(pipeline, params, cv=cv, n_iter=100, n_jobs=1, verbose=2)
    search.fit(X_search, y_search)

    print(search.best_score_)
    print(search.best_params_)

if __name__ == '__main__':
    main()
