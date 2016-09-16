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

from tqdm import tqdm
from model import Model

from sklearn.utils import shuffle

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

    num_features = len(feature_cols)
    features = tf.placeholder(tf.float32, shape=[None, num_features], name='features')
    targets = tf.placeholder(tf.int32, shape=[None], name='targets')

    with tf.variable_scope('model'):
        train_model = Model(features, targets, is_training=True)

    with tf.variable_scope('model', reuse=True):
        test_model = Model(features, targets, is_training=False)

    summary_op = tf.merge_all_summaries()
    logdir = 'logs/{}'.format(int(time.time()))
    supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None)
    with supervisor.managed_session() as sess:
        summary_writer = tf.train.SummaryWriter(logdir, graph=sess.graph)

        print('Training model with {} parameters...'.format(train_model.num_parameters))
        optimize_d, optimize_g = True, True
        with tqdm(total=FLAGS.num_epochs) as pbar:
            for epoch in range(FLAGS.num_epochs):
                X_train_epoch, y_train_epoch = shuffle(X_train, y_train)
                num_batches = len(y_train_epoch) // FLAGS.batch_size

                losses_d, losses_g = [], []
                for batch_index in range(num_batches):
                    batch_start = batch_index * FLAGS.batch_size
                    batch_end = batch_start + FLAGS.batch_size

                    X_train_batch = X_train_epoch[batch_start:batch_end]
                    y_train_batch = y_train_epoch[batch_start:batch_end]

                    if optimize_d:
                        _, loss_d = sess.run([
                            train_model.train_step_d,
                            train_model.loss_d,
                        ], feed_dict={
                            features: X_train_batch,
                            targets: y_train_batch,
                        })
                    else:
                        loss_d = sess.run(train_model.loss_d, feed_dict={
                            features: X_train_batch,
                            targets: y_train_batch,
                        })

                    if optimize_g:
                        _, loss_g = sess.run([
                            train_model.train_step_g,
                            train_model.loss_g,
                        ], feed_dict={
                            features: X_train_batch,
                            targets: y_train_batch,
                        })
                    else:
                        loss_g = sess.run(train_model.loss_g, feed_dict={
                            features: X_train_batch,
                            targets: y_train_batch,
                        })

                    losses_d.append(loss_d)
                    losses_g.append(loss_g)

                loss_train_d = np.mean(losses_d)
                loss_train_g = np.mean(losses_g)

                summary_str = sess.run(summary_op, feed_dict={
                    features: X_valid,
                    targets: y_valid,
                })

                optimize_d = epoch % 2 == 0
                optimize_g = True

                if not optimize_d and not optimize_g:
                    optimize_d = True
                    optimize_g = True

                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

                pbar.set_description('[{}] loss_train_d ({}): {:.8f}, loss_train_g ({}): {:.8f}'.format(epoch, optimize_d, loss_train_d, optimize_g, loss_train_g))
                pbar.update()

        summary_writer.close()

        loss_valid_d, loss_valid_g, summary_str = sess.run([
            test_model.loss_d,
            test_model.loss_g,
            summary_op,
        ], feed_dict={
            features: X_valid,
            targets: y_valid,
        })
        print('Validation loss (d): {:.8f}, loss (g): {:.8f}'.format(loss_valid_d, loss_valid_g))

        z_train = sess.run(test_model.z, feed_dict={ features: X_train })
        z_valid = sess.run(test_model.z, feed_dict={ features: X_valid })
        z_test = sess.run(test_model.z, feed_dict={ features: X_test })

        np.savez('data/adversarial.npz', z_train=z_train, z_valid=z_valid, z_test=z_test)

if __name__ == "__main__":
    tf.app.run()
