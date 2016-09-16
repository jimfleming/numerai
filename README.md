# Numerai Experiments

Folder structure:

- ensemble.py - combines multiple predictions using geometric mean
- fit_tsne.py - uses [this t-SNE implementation](https://github.com/danielfrg/tsne) for 2D embedding (does not work in 3D)
- search_params.py - uses `RandomSearchCV` for hyperparameter search
- tpot_test.py - runs [tpot](https://github.com/rhiever/tpot) over the data
- tpot_pipeline.py - best tpot model
- notebooks/ - contains Jupyter notebooks
- bh_tsne/ - is the original C++ t-SNE implementation with scripts for converting the csvs to the format the binary expects
- models/ - various model implementations

        - adverarial/ - generative adversarial model that saves the learned features for each sample
        - autoencoder/ - simple autoencoder with regular and denoising variants (also saves learned features)
        - classifier/ - simple neural network classifier
        - pairwise/ - pairwise model implementation described in the blog post
        - pipeline/ - various scikit-learn models

                - estimators.py - custom wrappers around `KernelPCA` and `Isomap` that fit on a small portion of the training samples to avoid memory errors
                - transformers.py - contains `ItemSelector` which allows for selecting data by a key when building pipelines ([source](http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html))
                - fm.py - factorization machines
                - lr.py - logistic regression with t-SNE features
                - pairwise.py - sklearn variant of the pairwise model
                - simple.py - simple logistic regression with polynomial features
