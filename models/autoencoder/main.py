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
from sklearn.cross_validation import train_test_split

from tqdm import tqdm
from model import Model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 30, "")
tf.app.flags.DEFINE_integer('batch_size', 128, "")
tf.app.flags.DEFINE_boolean('denoise', True, "")

if FLAGS.denoise:
    print('Denoising!')
else:
    print('NOT denoising!')

def main(_):
    df_train = pd.read_csv('data/train_data.csv')
    df_valid = pd.read_csv('data/valid_data.csv')
    df_test = pd.read_csv('data/test_data.csv')

    feature_cols = list(df_train.columns)[:-1]

    X_train = df_train[feature_cols].values
    X_valid = df_valid[feature_cols].values
    X_test = df_test[feature_cols].values

    num_features = len(feature_cols)
    features = tf.placeholder(tf.float32, shape=[None, num_features], name='features')

    with tf.variable_scope('model'):
        train_model = Model(features, denoise=FLAGS.denoise, is_training=True)

    with tf.variable_scope('model', reuse=True):
        test_model = Model(features, denoise=FLAGS.denoise, is_training=False)

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
                X_train_epoch = shuffle(X_train)
                num_batches = len(X_train_epoch) // FLAGS.batch_size


                losses = []
                for batch_index in range(num_batches):
                    batch_start = batch_index * FLAGS.batch_size
                    batch_end = batch_start + FLAGS.batch_size

                    X_train_batch = X_train_epoch[batch_start:batch_end]

                    _, loss = sess.run([
                        train_model.train_step,
                        train_model.loss,
                    ], feed_dict={
                        features: X_train_batch,
                    })
                    losses.append(loss)
                loss_train = np.mean(losses)

                loss_valid, summary_str = sess.run([
                    test_model.loss,
                    summary_op,
                ], feed_dict={
                    features: X_valid,
                })
                if best is None or loss_valid < best:
                    best = loss_valid
                    wait = 0
                else:
                    wait += 1
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()
                pbar.set_description('[{}] loss (train): {:.8f}, loss (valid): {:.8f} [best: {:.8f}, wait: {}]' \
                    .format(epoch, loss_train, loss_valid, best, wait))
                pbar.update()

        summary_writer.close()

        loss_valid = sess.run(test_model.loss, feed_dict={
            features: X_valid,
        })
        print('Validation loss: {}'.format(loss_valid))

        z_train = sess.run(test_model.z, feed_dict={ features: X_train })
        z_valid = sess.run(test_model.z, feed_dict={ features: X_valid })
        z_test = sess.run(test_model.z, feed_dict={ features: X_test })

        if FLAGS.denoise:
            np.savez('data/denoising.npz', z_train=z_train, z_valid=z_valid, z_test=z_test)
        else:
            np.savez('data/autoencoder.npz', z_train=z_train, z_valid=z_valid, z_test=z_test)

if __name__ == "__main__":
    tf.app.run()
