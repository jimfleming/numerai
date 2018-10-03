from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class Model(object):

    def __init__(self, features, denoise, is_training):
        sigmoid_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        weights_reg = tf.contrib.layers.l2_regularizer(1e-3)

        if denoise:
            _, features_variance = tf.nn.moments(features, axes=[0])
            features_stddev = tf.sqrt(features_variance)
            features_shape = tf.shape(features)
            features_noise = features + tf.random_normal(features_shape, stddev=features_stddev * 0.1)
        else:
            features_noise = features

        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = { 'is_training': is_training }

        h0 = tf.contrib.layers.fully_connected(
            inputs=features_noise,
            num_outputs=32,
            activation_fn=tf.sigmoid,
            weights_initializer=sigmoid_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        h1 = tf.contrib.layers.fully_connected(
            inputs=h0,
            num_outputs=24,
            activation_fn=tf.sigmoid,
            weights_initializer=sigmoid_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        self.z = h2 = tf.contrib.layers.fully_connected(
            inputs=h1,
            num_outputs=16,
            activation_fn=tf.sigmoid,
            weights_initializer=sigmoid_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        h3 = tf.contrib.layers.fully_connected(
            inputs=h2,
            num_outputs=24,
            activation_fn=tf.sigmoid,
            weights_initializer=sigmoid_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        h4 = tf.contrib.layers.fully_connected(
            inputs=h3,
            num_outputs=32,
            activation_fn=tf.sigmoid,
            weights_initializer=sigmoid_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        reconstruction = tf.contrib.layers.fully_connected(
            inputs=h4,
            num_outputs=45,
            activation_fn=tf.sigmoid,
            weights_initializer=sigmoid_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        self.loss = tf.reduce_mean(tf.squared_difference(features, reconstruction), name='loss')
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
                clip_gradients=1.0,
                optimizer=optimizer)
                #moving_average_decay=None)

    @property
    def num_parameters(self):
        return sum([np.prod(tvar.get_shape().as_list()) for tvar in tf.trainable_variables()])
