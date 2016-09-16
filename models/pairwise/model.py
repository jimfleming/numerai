from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class Model(object):

    def __init__(self, features_L, features_R, targets, is_training):
        with tf.variable_scope('feature_extractor'):
            embedding_L = self.feature_extractor(features_L, is_training)

        with tf.variable_scope('feature_extractor', reuse=True):
            embedding_R = self.feature_extractor(features_R, is_training)

        embedding = tf.concat(1, [embedding_L, embedding_R])
        logits = self.classifier(embedding, is_training)
        self.predictions = tf.nn.softmax(logits)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
        self.loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.contrib.layers.summarize_tensor(self.loss)
        tf.contrib.losses.add_loss(self.loss)

        self.total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

        # setup learning
        if is_training:
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.learning_rate = 1e-4

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_step = tf.contrib.layers.optimize_loss(self.total_loss, self.global_step, \
                learning_rate=self.learning_rate,
                clip_gradients=None,
                gradient_noise_scale=None,
                optimizer=optimizer,
                moving_average_decay=None)

    def feature_extractor(self, features, is_training):
        relu_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN')
        weights_reg = tf.contrib.layers.l2_regularizer(1e-3)

        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = { 'is_training': is_training }

        h0 = tf.contrib.layers.fully_connected(
            inputs=features,
            num_outputs=16,
            activation_fn=tf.nn.relu,
            weights_initializer=relu_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        h1 = tf.contrib.layers.fully_connected(
            inputs=h0,
            num_outputs=8,
            activation_fn=tf.nn.relu,
            weights_initializer=relu_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        return h1

    def classifier(self, features, is_training):
        relu_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN')
        softmax_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        weights_reg = tf.contrib.layers.l2_regularizer(1e-3)

        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = { 'is_training': is_training }

        h1 = tf.contrib.layers.fully_connected(
            inputs=features,
            num_outputs=16,
            activation_fn=tf.nn.relu,
            weights_initializer=relu_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        h2 = tf.contrib.layers.fully_connected(
            inputs=h1,
            num_outputs=2,
            activation_fn=None,
            weights_initializer=softmax_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        return h2

    @property
    def num_parameters(self):
        return sum([np.prod(tvar.get_shape().as_list()) for tvar in tf.trainable_variables()])
