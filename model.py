import os
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.data import Dataset, Iterator


#### NETWORK
def create_network_dummy(input_tensor, target_tensor, n_classes):

    net = tf.layers.conv2d(input_tensor, 5, 3, padding='same', use_bias=False)
    net = tf.nn.relu(net)
    net = tf.layers.average_pooling2d(net, 32, 1)
    net = tf.reshape(net, [-1, 5])
    logits = tf.layers.dense(net, n_classes)

    return logits


def create_network(input_tensor, target_tensor, network_name, network_args):

    if network_name == 'dummy':
        network_output_tensor = create_network_dummy(input_tensor, target_tensor, *network_args)
    else:
        raise Exception("Invalid network name: {}".format(network_name))

    return network_output_tensor


#### LOSS
def create_loss(target_tensor, network_output_tensor, loss_name):

    if loss_name == 'sparse_softmax_cross_entropy':
        loss_tensor = tf.losses.sparse_softmax_cross_entropy(target_tensor, network_output_tensor)
    else:
        raise Exception("Invalid loss name: {}".format(loss_name))

    return loss_tensor


#### EPOCH
def create_epoch():

    epoch = tf.get_variable(
        'epoch',
        [],
        initializer=tf.initializers.constant(-1),
        trainable=False)

    epoch_increment_asgn = tf.assign_add(epoch, tf.constant(1.0))

    return epoch, epoch_increment_asgn


#### LEARNING RATE SCHEDULER
def create_learning_rate_scheduler(epoch, lr_init, lr_decay, lr_schedule):

    lr = tf.get_variable(
        'learning_rate',
        [],
        initializer=tf.initializers.constant(lr_init),
        trainable=False)

    lr_init = tf.constant(lr_init, name='learning_rate_init', dtype=tf.float32)
    lr_decay = tf.constant(lr_decay, name='learning_rate_decay', dtype=tf.float32)
    lr_schedule = tf.constant(lr_schedule, name='learning_rate_schedule', dtype=tf.float32)

    lr_new_value = lr_init * lr_decay ** tf.reduce_sum(tf.cast(lr_schedule <= epoch, tf.float32))
    lr_scheduler = tf.assign(lr, lr_new_value)

    return lr_scheduler


#### OPTIMIZER
def create_optimizer(loss_tensor, optimizer_name, lr_scheduler, momentum, nesterov):

    if optimizer_name == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_scheduler, momentum, use_nesterov=nesterov, name='sgd')
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(lr_scheduler, name='adam')
    else:
        raise Exception("Invalid optimzer name: {}".format(optimizer_name))

    optimizer_op = optimizer.minimize(loss_tensor)

    return optimizer_op


#### STATS

def create_stats_error_rate(target_tensor, network_output_tensor):

    with tf.variable_scope('error_rate'):

        # number of errors
        n_errors = tf.get_variable(
            'n_errors',
            [],
            initializer=tf.initializers.zeros(),
            trainable=False,
            collections=['stats'])
        n_errors_cur = tf.argmax(network_output_tensor, 1)
        n_errors_cur = tf.not_equal(n_errors_cur, tf.reshape(target_tensor, [-1]))
        n_errors_cur = tf.reduce_sum(tf.cast(n_errors_cur, tf.float32))
        n_errors_update_asgn = tf.assign_add(n_errors, n_errors_cur)

        # number of observations
        n_observations = tf.get_variable(
            'n_observations',
            [],
            initializer=tf.initializers.zeros(),
            trainable=False,
            collections=['stats'])
        n_observations_cur = tf.cast(tf.shape(target_tensor)[0], tf.float32)
        n_observations_update_asgn = tf.assign_add(n_observations, n_observations_cur)

        # error rate
        error_rate_tensor = n_errors / n_observations
        error_rate_summary_protobuf = tf.summary.scalar('error_rate', error_rate_tensor)

        # callbacks
        error_rate_reset = tf.variables_initializer([n_errors, n_observations])
        error_rate_update = [n_errors_update_asgn, n_observations_update_asgn]
        error_rate_save = tf.summary.merge([error_rate_summary_protobuf])

        error_rate = [error_rate_reset, error_rate_update, error_rate_save]

    return error_rate


def create_stats(stats_train_name, stats_test_name, target_tensor, network_output_tensor):

    stats_train_reset = []
    stats_train_update = []
    stats_train_save = []
    if stats_train_name == 'error_rate':
        error_rate = create_stats_error_rate(target_tensor, network_output_tensor)
        stats_train_reset.append(error_rate[0])
        stats_train_update.append(error_rate[1])
        stats_train_save.append(error_rate[2])
    else:
        raise Exception("Invalid stats train name: {}".format(stats_train_name))

    stats_train_save = tf.summary.merge(stats_train_save)

    stats_train = {
        'reset': stats_train_reset,
        'update': stats_train_update,
        'save': stats_train_save}

    stats_test = stats_train

    return stats_train, stats_test


#### INITIALIZE
def initialize(
        input_tensor,
        target_tensor,
        network_name,
        network_args,
        loss_name,
        stats_train_name,
        stats_test_name,
        optimizer_name,
        momentum,
        nesterov,
        weight_decay,
        lr_init,
        lr_decay,
        lr_schedule):


    # create network
    with tf.variable_scope('network'):
        # TODO add weight decay
        network_output_tensor = create_network(input_tensor, target_tensor, network_name, network_args)

    # create loss
    with tf.variable_scope('loss'):
        loss_tensor = create_loss(target_tensor, network_output_tensor, loss_name)

    # create epoch
    with tf.variable_scope('epoch'):
        epoch_tensor, epoch_increment_asgn = create_epoch()

    # create learning rate scheduler
    with tf.variable_scope('learning_rate'):
        lr_scheduler = create_learning_rate_scheduler(epoch_tensor, lr_init, lr_decay, lr_schedule)

    # create optimizer
    with tf.variable_scope('optimizer'):
        optimizer_op = create_optimizer(loss_tensor, optimizer_name, lr_scheduler, momentum, nesterov)

    # create stats
    with tf.variable_scope('stats'):
        stats_train, stats_test = create_stats(
            stats_train_name, stats_test_name, target_tensor, network_output_tensor)

    # create final train steps
    train_begin = stats_train['reset'] + [epoch_increment_asgn]
    train_step = [optimizer_op] + stats_train['update']
    train_summary = stats_train['save']

    # create final test steps
    test_begin = stats_test['reset']
    test_step = stats_test['update']
    test_summary = stats_test['save']

    # model
    model = [
        epoch_tensor,
        train_begin,
        train_step,
        train_summary,
        test_begin,
        test_step,
        test_summary,
    ]

    return model

