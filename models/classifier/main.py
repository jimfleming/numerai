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

import tensorflow as tf
tf.set_random_seed(67)

from sklearn.utils import shuffle
from sklearn.metrics import log_loss, roc_auc_score

from tqdm import tqdm, trange
from model import Model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 30, "")
tf.app.flags.DEFINE_integer('batch_size', 128, "")

def main(_):
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

    tsne_data_5 = np.load('data/tsne_2d_5p_poly.npz')
    tsne_data_15 = np.load('data/tsne_2d_15p_poly.npz')
    tsne_data_10 = np.load('data/tsne_2d_10p_poly.npz')
    tsne_data_20 = np.load('data/tsne_2d_20p_poly.npz')
    tsne_data_30 = np.load('data/tsne_2d_30p_poly.npz')
    tsne_data_40 = np.load('data/tsne_2d_40p_poly.npz')
    tsne_data_50 = np.load('data/tsne_2d_50p_poly.npz')

    X_train_concat = np.concatenate([
        X_train,
        tsne_data_5['train'],
        tsne_data_15['train'],
        tsne_data_50['train'],
    ], axis=1)
    X_valid_concat = np.concatenate([
        X_valid,
        tsne_data_5['valid'],
        tsne_data_15['valid'],
        tsne_data_50['valid'],
    ], axis=1)
    X_test_concat = np.concatenate([
        X_test,
        tsne_data_5['test'],
        tsne_data_15['test'],
        tsne_data_50['test'],
    ], axis=1)

    num_features = X_train_concat.shape[1]
    features = tf.placeholder(tf.float32, shape=[None, num_features], name='features')
    targets = tf.placeholder(tf.int32, shape=[None], name='targets')

    with tf.variable_scope('model'):
        train_model = Model(features, targets, is_training=True)

    with tf.variable_scope('model', reuse=True):
        test_model = Model(features, targets, is_training=False)

    best = None
    wait = 0

    summary_op = tf.merge_all_summaries()
    logdir = 'logs/{}'.format(int(time.time()))
    supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None)
    with supervisor.managed_session() as sess:
        summary_writer = tf.train.SummaryWriter(logdir, graph=sess.graph)

        print('Training model with {} parameters...'.format(train_model.num_parameters))
        with tqdm(total=FLAGS.num_epochs) as pbar:
            for epoch in range(FLAGS.num_epochs):
                X_train_epoch, y_train_epoch = shuffle(X_train_concat, y_train)

                num_batches = len(y_train_epoch) // FLAGS.batch_size

                losses = []
                for batch_index in range(num_batches):
                    batch_start = batch_index * FLAGS.batch_size
                    batch_end = batch_start + FLAGS.batch_size

                    X_train_batch = X_train_epoch[batch_start:batch_end]
                    y_train_batch = y_train_epoch[batch_start:batch_end]

                    _, loss = sess.run([
                        train_model.train_step,
                        train_model.loss,
                    ], feed_dict={
                        features: X_train_batch,
                        targets: y_train_batch,
                    })
                    losses.append(loss)
                loss_train = np.mean(losses)

                loss_valid, summary_str = sess.run([
                    test_model.loss,
                    summary_op
                ], feed_dict={
                    features: X_valid_concat,
                    targets: y_valid,
                })

                if best is None or loss_valid < best:
                    best = loss_valid
                    wait = 0
                else:
                    wait += 1

                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

                pbar.set_description('[{}] (train) loss: {:.8f}, (valid) loss: {:.8f} [best: {:.8f}, wait: {}]' \
                    .format(epoch, loss_train, loss_valid, best, wait))
                pbar.update()

        summary_writer.close()

        p_valid = sess.run(test_model.predictions, feed_dict={
            features: X_valid_concat,
        })
        loss = log_loss(y_valid, p_valid[:,1])
        auc = roc_auc_score(y_valid, p_valid[:,1])
        print('Validation Prediction Loss: {}, AUC: {}'.format(loss, auc))

        p_test = sess.run(test_model.predictions, feed_dict={
            features: X_test_concat,
        })
        df_pred = pd.DataFrame({
            't_id': df_test['t_id'],
            'probability': p_test[:,1]
        })
        csv_path = 'predictions/predictions_{}_{}.csv'.format(int(time.time()), loss)
        df_pred.to_csv(csv_path, columns=('t_id', 'probability'), index=None)
        print('Saved: {}'.format(csv_path))

if __name__ == "__main__":
    tf.app.run()
