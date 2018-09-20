import numpy as np
import tensorflow as tf


def create_dummy(input_tensor, target_tensor, n_classes):

    net = tf.layers.conv2d(input_tensor, 5, 3, padding='same', use_bias=False)
    net = tf.nn.relu(net)
    net = tf.layers.average_pooling2d(net, 32, 1)
    net = tf.reshape(net, [-1, 5])
    logits = tf.layers.dense(net, n_classes)

    return logits


def create(input_tensor, target_tensor, network_name, network_args):

    if network_name == 'dummy':
        network_output_tensor = create_dummy(input_tensor, target_tensor, *network_args)
    else:
        raise Exception("Invalid network name: {}".format(network_name))

    return network_output_tensor

