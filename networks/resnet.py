import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def create_conv2d(input_tensor, n_filters, kernel_size, stride, weight_decay, name):
    h = input_tensor
    h = tf.layers.conv2d(
        h, n_filters, kernel_size, stride, padding='same', use_bias=False,
        kernel_initializer=keras.initializers.he_normal(),
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        name=name)
    return h


def basic_block(x, nf_in, nf_out, stride, training_mode, weight_decay, name):
    with tf.variable_scope(name):
        h = x
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_1')
        h = tf.nn.relu(h, name='relu_1')
        if (stride != 1) or (nf_in != nf_out):
            s = create_conv2d(h, nf_out, 1, stride, weight_decay, 'skip')
        else:
            s = x
        h = create_conv2d(h, nf_out, 3, stride, weight_decay, 'conv2d_1')
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_2')
        h = tf.nn.relu(h, name='relu_2')
        h = create_conv2d(h, nf_out, 3, 1, weight_decay, 'conv2d_2')

        h = h + s

    return h


def bottleneck_block(x, nf_in, nf_out, stride, training_mode, weight_decay, name):
    with tf.variable_scope(name):
        h = x
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_1')
        h = tf.nn.relu(h, name='relu_1')
        if (stride != 1) or (nf_in != nf_out):
            s = create_conv2d(h, nf_out, 1, stride, weight_decay, 'skip')
        else:
            s = x
        h = create_conv2d(h, int(nf_out / 4), 1, 1, weight_decay, 'conv2d_1')
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_2')
        h = tf.nn.relu(h, name='relu_2')
        h = create_conv2d(h, int(nf_out / 4), 3, stride, weight_decay, 'conv2d_2')
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_3')
        h = tf.nn.relu(h, name='relu_3')
        h = create_conv2d(h, nf_out, 1, 1, weight_decay, 'conv2d_3')

        h = h + s

    return h


def create_block_group(input_tensor, block, n_blocks, nf_in, nf_out, stride, training_mode, weight_decay, name):
    x = input_tensor
    with tf.variable_scope(name):
        x = block(x, nf_in, nf_out, stride, training_mode, weight_decay, 'block_1')
        for i in range(n_blocks - 1):
            x = block(x, nf_out, nf_out, 1, training_mode, weight_decay, 'block_{}'.format(i+2))

    return x


def create_graph(input_tensor, training_mode, weight_decay, n_classes, widen, block, n_blocks):

    x = input_tensor

    with tf.variable_scope('features'):
        x = create_conv2d(x, 16, 3, 1, weight_decay, 'conv1')
        x = create_block_group(x, block, n_blocks[0], 16, 16 * widen, 1, training_mode, weight_decay, 'group_1')
        x = create_block_group(x, block, n_blocks[1], 16 * widen, 32 * widen, 2, training_mode, weight_decay, 'group_2')
        x = create_block_group(x, block, n_blocks[2], 32 * widen, 64 * widen, 2, training_mode, weight_decay, 'group_3')
        x = tf.layers.batch_normalization(x, training=training_mode, name='bn_last')
        x = tf.nn.relu(x, name='relu_last')
        x = keras.layers.GlobalAveragePooling2D()(x)

    with tf.variable_scope('classifier'):
        logits = tf.layers.dense(x, n_classes,
                                 kernel_initializer=keras.initializers.he_uniform(),
                                 kernel_regularizer=keras.regularizers.l2(weight_decay))

    return logits


def create(input_tensor, training_mode, weight_decay, n_classes, depth, widen):
    if depth % 6 != 2:
        raise ValueError('Depth must be 6n + 2. Depth was: ', depth)

    n_blocks = [int((depth - 2) / 6)] * 3
    logits = create_graph(input_tensor, training_mode, weight_decay, n_classes, widen, basic_block, n_blocks)

    return logits

