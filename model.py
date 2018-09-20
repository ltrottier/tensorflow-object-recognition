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

    # create loss tensor
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
    tf.add_to_collection("train_begin", epoch_increment_asgn)

    return epoch


#### LEARNING RATE SCHEDULER
def create_learning_rate_scheduler(epoch_tensor, lr_init, lr_decay, lr_schedule):

    lr_tensor = tf.get_variable(
        'learning_rate',
        [],
        initializer=tf.initializers.constant(lr_init),
        trainable=False)

    lr_init_tensor = tf.constant(lr_init, name='learning_rate_init', dtype=tf.float32)
    lr_decay_tensor = tf.constant(lr_decay, name='learning_rate_decay', dtype=tf.float32)
    lr_schedule_tensor = tf.constant(lr_schedule, name='learning_rate_schedule', dtype=tf.float32)
    lr_new_value = lr_init_tensor * lr_decay_tensor ** tf.reduce_sum(tf.cast(lr_schedule_tensor <= epoch_tensor, tf.float32))
    lr_scheduler = tf.assign(lr_tensor, lr_new_value)
    tf.add_to_collection("train_begin", lr_scheduler)

    return lr_tensor


#### OPTIMIZER
def create_optimizer(loss_tensor, optimizer_name, lr_scheduler, momentum, nesterov):

    if optimizer_name == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_scheduler, momentum, use_nesterov=nesterov, name='sgd')
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(lr_scheduler, name='adam')
    else:
        raise Exception("Invalid optimzer name: {}".format(optimizer_name))

    optimizer_op = optimizer.minimize(loss_tensor)
    tf.add_to_collection("train_step", optimizer_op)

    return optimizer_op


#### STATS

def create_stats_n_observations(network_output_tensor, modes):
    # number of observations
    with tf.variable_scope('n_observations'):
        n_observations = tf.get_variable(
            'variable',
            [],
            initializer=tf.initializers.zeros(),
            trainable=False,
            collections=['stats'])
        n_observations_cur = tf.cast(tf.shape(network_output_tensor)[0], tf.float32)
        n_observations_update_asgn = tf.assign_add(n_observations, n_observations_cur)
        n_observations_init = tf.variables_initializer([n_observations])
        n_observations_summary_protobuf = tf.summary.scalar('summary', n_observations)

    for mode in modes:
        tf.add_to_collection("{}_begin".format(mode), n_observations_init)
        tf.add_to_collection("{}_step".format(mode), n_observations_update_asgn)
        tf.add_to_collection("{}_summary".format(mode), n_observations_summary_protobuf)


def create_stats_error_rate(target_tensor, network_output_tensor, modes):
    # number of observations
    n_observations = tf.get_variable('n_observations/variable')

    with tf.variable_scope('error_rate'):
        # number of errors
        n_errors = tf.get_variable(
            'n_errors',
            [],
            initializer=tf.initializers.zeros(),
            trainable=False,
            collections=['stats'])
        n_errors_init = tf.variables_initializer([n_errors])
        n_errors_cur = tf.argmax(network_output_tensor, 1)
        n_errors_cur = tf.not_equal(n_errors_cur, tf.reshape(target_tensor, [-1]))
        n_errors_cur = tf.reduce_sum(tf.cast(n_errors_cur, tf.float32))
        n_errors_update_asgn = tf.assign_add(n_errors, n_errors_cur)

        # error rate
        error_rate_tensor = n_errors / n_observations
        error_rate_summary_protobuf = tf.summary.scalar('summary', error_rate_tensor)

        # collection
        for mode in modes:
            tf.add_to_collection("{}_begin".format(mode), n_errors_init)
            tf.add_to_collection("{}_step".format(mode), n_errors_update_asgn)
            tf.add_to_collection("{}_summary".format(mode), error_rate_summary_protobuf)


def create_stats_loss_average(network_output_tensor, loss_tensor, modes):

    # number of observations
    n_observations = tf.get_variable('n_observations/variable')

    with tf.variable_scope('loss_average'):
        # create loss variable
        loss_sum = tf.get_variable(
            'loss_sum',
            [],
            initializer=tf.initializers.constant(0),
            trainable=False)
        loss_sum_init = tf.variables_initializer([loss_sum])
        n_observations_cur = tf.cast(tf.shape(network_output_tensor)[0], tf.float32)
        loss_sum_update_asgn = tf.assign_add(loss_sum, loss_tensor * n_observations_cur)
        loss_average = loss_sum / n_observations
        loss_average_summary_protobuf = tf.summary.scalar('summary', loss_average)

        for mode in modes:
            tf.add_to_collection("{}_begin".format(mode), loss_sum_init)
            tf.add_to_collection("{}_step".format(mode), loss_sum_update_asgn)
            tf.add_to_collection("{}_summary".format(mode), loss_average_summary_protobuf)


def create_stats(stats_train_list, stats_test_list, target_tensor, network_output_tensor, loss_tensor):

    # since we compute stats per epoch, always create the stats for the number of observations
    stats_train_list = ['n_observations'] + stats_train_list
    stats_test_list = ['n_observations'] + stats_test_list

    def add_stats(name, target_tensor, network_output_tensor, loss_tensor, modes):
        if name == 'n_observations':
            create_stats_n_observations(network_output_tensor, modes)
        elif name == 'error_rate':
            create_stats_error_rate(target_tensor, network_output_tensor, modes)
        elif name == 'loss_average':
            create_stats_loss_average(network_output_tensor, loss_tensor, modes)
        else:
            raise Exception("Invalid stats name: {}, for modes: {}".format(name, modes))

    # train stats
    for stats_name in stats_train_list:
        modes = ['train', 'test'] if stats_name in stats_test_list else ['train']
        add_stats(stats_name, target_tensor, network_output_tensor, loss_tensor, modes)

    # test stats
    for stats_name in stats_test_list:
        if stats_name in stats_train_list:
            continue
        add_stats(stats_name, target_tensor, network_output_tensor, loss_tensor, ['test'])


#### INITIALIZE
def initialize(
        input_tensor,
        target_tensor,
        network_name,
        network_args,
        loss_name,
        stats_train_list,
        stats_test_list,
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
        epoch_tensor = create_epoch()

    # create learning rate scheduler
    with tf.variable_scope('learning_rate'):
        lr_tensor = create_learning_rate_scheduler(epoch_tensor, lr_init, lr_decay, lr_schedule)

    # create optimizer
    with tf.variable_scope('optimizer'):
        create_optimizer(loss_tensor, optimizer_name, lr_tensor, momentum, nesterov)

    # create stats
    with tf.variable_scope('stats', reuse=tf.AUTO_REUSE):
        create_stats(stats_train_list, stats_test_list, target_tensor, network_output_tensor, loss_tensor)


    train_begin = tf.get_collection("train_begin")
    train_step = tf.get_collection("train_step")
    train_summary = tf.summary.merge(tf.get_collection("train_summary"))
    train_end = tf.get_collection("train_end")

    test_begin = tf.get_collection("test_begin")
    test_step = tf.get_collection("test_step")
    test_summary = tf.summary.merge(tf.get_collection("test_summary"))
    test_end = tf.get_collection("test_end")

    # create final train steps
    #train_begin = stats_train['reset'] + [epoch_increment_asgn] + [lr_scheduler]
    #train_step = [optimizer_op] + stats_train['update']
    #train_summary = stats_train['save']

    # create final test steps
    #test_begin = stats_test['reset']
    #test_step = stats_test['update']
    #test_summary = stats_test['save']

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

