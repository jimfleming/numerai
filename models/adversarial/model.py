from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class Model(object):

    def __init__(self, features, targets, is_training):
        batch_size = tf.shape(targets)[0]

        z = tf.random_normal([batch_size, 10], mean=0.0, stddev=1.0, dtype=tf.float32, name='z')
        z.set_shape([None, 10])

        with tf.variable_scope('generator') as scope:
            G = self.generator(z, is_training)

        with tf.variable_scope('discriminator'):
            D, embedding = self.discriminator(features, is_training)
            self.z = embedding

        with tf.variable_scope('discriminator', reuse=True):
            D_, embedding_ = self.discriminator(G, is_training)

        self.loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D, tf.ones_like(D)))
        self.loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_, tf.zeros_like(D_)))
        self.loss_d = self.loss_real + self.loss_fake
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_, tf.ones_like(D_)))

        if is_training:
            tf.contrib.layers.summarize_tensor(self.loss_real, tf.get_variable_scope().name + '/loss_real')
            tf.contrib.layers.summarize_tensor(self.loss_fake, tf.get_variable_scope().name + '/loss_fake')
            tf.contrib.layers.summarize_tensor(self.loss_d, tf.get_variable_scope().name + '/loss_d')
            tf.contrib.layers.summarize_tensor(self.loss_g, tf.get_variable_scope().name + '/loss_g')

        # setup learning
        if is_training:
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            optimizer_d = tf.train.AdamOptimizer(learning_rate=1e-4)
            optimizer_g = tf.train.AdamOptimizer(learning_rate=1e-4)

            tvars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/discriminator')
            tvars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/generator')

            regularizer = tf.contrib.layers.l2_regularizer(1e-3)
            reg_d = tf.contrib.layers.apply_regularization(regularizer, weights_list=tvars_d)
            reg_g = tf.contrib.layers.apply_regularization(regularizer, weights_list=tvars_g)

            self.train_step_d = optimizer_d.minimize(self.loss_d + reg_d,
                global_step=self.global_step,
                var_list=tvars_d)
            self.train_step_g = optimizer_g.minimize(self.loss_g + reg_g,
                global_step=self.global_step,
                var_list=tvars_g)

    def discriminator(self, features, is_training):
        softmax_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        weights_reg = None # tf.contrib.layers.l2_regularizer(1e-3)

        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = { 'is_training': is_training }

        h0 = tf.contrib.layers.fully_connected(
            inputs=features,
            num_outputs=32,
            activation_fn=tf.sigmoid,
            weights_initializer=softmax_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        h1 = tf.contrib.layers.fully_connected(
            inputs=h0,
            num_outputs=24,
            activation_fn=tf.sigmoid,
            weights_initializer=softmax_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        h2 = tf.contrib.layers.fully_connected(
            inputs=h1,
            num_outputs=16,
            activation_fn=tf.sigmoid,
            weights_initializer=softmax_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)

        logits = tf.contrib.layers.fully_connected(
            inputs=h2,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=softmax_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)
        return logits, h2

    def generator(self, z, is_training):
        sigmoid_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        weights_reg = None # tf.contrib.layers.l2_regularizer(1e-3)

        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = { 'is_training': is_training }

        h0 = tf.contrib.layers.fully_connected(
            inputs=z,
            num_outputs=16,
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
        h2 = tf.contrib.layers.fully_connected(
            inputs=h1,
            num_outputs=32,
            activation_fn=tf.sigmoid,
            weights_initializer=sigmoid_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)
        h3 = tf.contrib.layers.fully_connected(
            inputs=h2,
            num_outputs=21,
            activation_fn=tf.sigmoid,
            weights_initializer=sigmoid_init,
            weights_regularizer=weights_reg,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params)
        return h3

    @property
    def num_parameters(self):
        return sum([np.prod(tvar.get_shape().as_list()) for tvar in tf.trainable_variables()])
