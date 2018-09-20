import numpy as np
import tensorflow as tf

import stats
import networks


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
        initializer=tf.initializers.constant(0),
        trainable=False)

    epoch_increment_asgn = tf.assign_add(epoch, tf.constant(1.0))
    tf.add_to_collection("test_end", epoch_increment_asgn)

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
        network_output_tensor = networks.create(input_tensor, target_tensor, network_name, network_args)

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
        stats.create_from_list(stats_train_list, stats_test_list, target_tensor, network_output_tensor, loss_tensor)


    # create train calls
    train_begin = tf.get_collection("train_begin")
    train_step = tf.get_collection("train_step")
    train_summary = tf.summary.merge(tf.get_collection("train_summary"))
    train_end = tf.get_collection("train_end")

    # create test calls
    test_begin = tf.get_collection("test_begin")
    test_step = tf.get_collection("test_step")
    test_summary = tf.summary.merge(tf.get_collection("test_summary"))
    test_end = tf.get_collection("test_end")

    # model
    model = [
        epoch_tensor,
        train_begin,
        train_step,
        train_summary,
        train_end,
        test_begin,
        test_step,
        test_summary,
        test_end,
    ]

    return model

