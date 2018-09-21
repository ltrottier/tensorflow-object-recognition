# global
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# local
import networks.resnet as resnet

def create_dummy(input_tensor, training_mode, weight_decay, n_classes):

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
            dtype=tf.bool,
            trainable=False,
            initializer=tf.initializers.constant(False))

        training_mode_int = tf.cast(training_mode, tf.int64)
        training_mode_summary_protobuf = tf.summary.scalar('summary', training_mode_int)
        tf.add_to_collection('train_summary', training_mode_summary_protobuf)
        tf.add_to_collection('test_summary', training_mode_summary_protobuf)

        training_mode_asign_true = tf.assign(training_mode, True, name='assign_to_true')
        tf.add_to_collection('train_begin', training_mode_asign_true)

        training_mode_asign_false = tf.assign(training_mode, False, name='assign_to_false')
        tf.add_to_collection('test_begin', training_mode_asign_false)

    return training_mode


def create_network_n_parameters():
    with tf.variable_scope('network_n_parameters'):
        n_parameters = tf.add_n([tf.size(v) for v in tf.trainable_variables()])
        n_parameters_summary_protobuf = tf.summary.scalar('summary', n_parameters)
        tf.add_to_collection('train_summary', n_parameters_summary_protobuf)


def create(input_tensor, target_tensor, network_name, network_args, weight_decay):

    with tf.variable_scope('network'):
        # create training mode
        training_mode = create_training_mode()

        # create network
        if network_name == 'dummy':
            network_output_tensor = create_dummy(input_tensor, training_mode, weight_decay, *network_args)
        elif network_name == 'resnet':
            network_output_tensor = resnet.create(input_tensor, training_mode, weight_decay, *network_args)
        else:
            raise Exception("Invalid network name: {}".format(network_name))

        # create network number of parameters
        create_network_n_parameters()

    return network_output_tensor

