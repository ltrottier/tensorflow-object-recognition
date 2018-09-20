import numpy as np
import tensorflow as tf


def create_dummy(input_tensor, target_tensor, n_classes):

    net = tf.layers.conv2d(input_tensor, 5, 3, padding='same', use_bias=False)
    net = tf.nn.relu(net)
    net = tf.layers.average_pooling2d(net, 32, 1)
    net = tf.reshape(net, [-1, 5])
    logits = tf.layers.dense(net, n_classes)

    return logits


def create_training_mode():

    with tf.variable_scope('training_mode'):
        training_mode = tf.get_variable(
            'variable',
            [],
            trainable=False,
            initializer=tf.initializers.constant(True))

        training_mode_int = tf.cast(training_mode, tf.int64)
        training_mode_summary_protobuf = tf.summary.scalar('summary', training_mode_int)
        tf.add_to_collection('train_summary', training_mode_summary_protobuf)
        tf.add_to_collection('test_summary', training_mode_summary_protobuf)

        training_mode_asign_true = tf.assign(training_mode, True)
        tf.add_to_collection('train_begin', training_mode_asign_true)

        training_mode_asign_false = tf.assign(training_mode, False)
        tf.add_to_collection('test_begin', training_mode_asign_false)

    return training_mode


def create(input_tensor, target_tensor, network_name, network_args):

    with tf.variable_scope('network'):
        # create training mode
        training_mode_tensor = create_training_mode()

        # TODO add weight decay

        if network_name == 'dummy':
            network_output_tensor = create_dummy(input_tensor, target_tensor, *network_args)
        else:
            raise Exception("Invalid network name: {}".format(network_name))

    return network_output_tensor

