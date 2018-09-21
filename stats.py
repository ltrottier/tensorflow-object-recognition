import numpy as np
import tensorflow as tf


def create_n_observations(network_output_tensor, modes):
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


def create_error_rate(target_tensor, network_output_tensor, modes):

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


def create_loss_average(network_output_tensor, loss_tensor, modes):

    # number of observations
    n_observations = tf.get_variable('n_observations/variable')

    with tf.variable_scope('loss_average'):
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


def create_input_image_visualization(input_tensor, modes):

    # input summary
    with tf.variable_scope('input_image_visualization'):
        input_image = tf.get_variable(
            'variable',
            [],
            initializer=tf.initializers.zeros(),
            trainable=False,
            validate_shape=False)

        first_batch = tf.get_variable(
            'condition',
            [],
            dtype=tf.bool,
            initializer=tf.initializers.constant(True),
            trainable=False)

        def true_fn():
            first_batch_asgn = tf.assign(first_batch, False)
            input_image_asgn = tf.assign(input_image, input_tensor, validate_shape=False)
            return [first_batch_asgn, input_image_asgn]

        def false_fn():
            return [first_batch, input_image]

        [first_batch_cond_asgn, input_image_cond_asgn] = tf.cond(tf.equal(first_batch, True), true_fn, false_fn)

        input_image_summary_protobuf = tf.summary.image('image', input_image, 10)
        first_batch_init = tf.variables_initializer([first_batch])

        for mode in modes:
            tf.add_to_collection("{}_begin".format(mode), first_batch_init)
            tf.add_to_collection("{}_step".format(mode), first_batch_cond_asgn)
            tf.add_to_collection("{}_step".format(mode), input_image_cond_asgn)
            tf.add_to_collection("{}_summary".format(mode), input_image_summary_protobuf)



def create_from_list(stats_train_list, stats_test_list, input_tensor, target_tensor, network_output_tensor, loss_tensor):

    # since we compute stats per epoch, always create the stats for the number of observations
    stats_train_list = ['n_observations'] + stats_train_list
    stats_test_list = ['n_observations'] + stats_test_list

    with tf.variable_scope('stats', reuse=tf.AUTO_REUSE):

        def add_stats(name, target_tensor, network_output_tensor, loss_tensor, modes):
            if name == 'n_observations':
                create_n_observations(network_output_tensor, modes)
            elif name == 'error_rate':
                create_error_rate(target_tensor, network_output_tensor, modes)
            elif name == 'loss_average':
                create_loss_average(network_output_tensor, loss_tensor, modes)
            elif name == 'input_image_visualization':
                create_input_image_visualization(input_tensor, modes)
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
