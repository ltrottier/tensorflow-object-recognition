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


def basic_block(x, n_filters, mode, training_mode, weight_decay, name):
    with tf.variable_scope(name):
        h = x
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_1')
        h = tf.nn.relu(h, name='relu_1')
        z = h
        h = create_conv2d(h, n_filters, 3, 1, weight_decay, 'conv2d_1')
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_2')
        h = tf.nn.relu(h, name='relu_2')

        if mode == 'downsample':
            h = create_conv2d(h, n_filters * 2, 3, 2, weight_decay, 'conv2d_2')
            s = create_conv2d(z, n_filters * 2, 1, 2, weight_decay, 'downsample')
        elif mode == 'enlarge':
            h = create_conv2d(h, n_filters * 2, 3, 1, weight_decay, 'conv2d_2')
            s = create_conv2d(z, n_filters * 2, 1, 1, weight_decay, 'enlarge')
        elif mode == 'normal':
            h = create_conv2d(h, n_filters, 3, 1, weight_decay, 'conv2d_2')
            s = x
        else:
            raise Exception("Invalid residual block mode: {}".format(mode))

        h = h + s

    return h

def bottleneck_block(x, n_filters, mode, training_mode, weight_decay, name):
    with tf.variable_scope(name):
        h = x
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_1')
        h = tf.nn.relu(h, name='relu_1')
        z = h

        if mode == 'downsample':
            stride = 2
            mult = 2
            s = create_conv2d(z, n_filters * 2, 1, 2, weight_decay, 'downsample')
        elif mode == 'enlarge':
            stride = 1
            mult = 2
            s = create_conv2d(z, n_filters * 2, 1, 1, weight_decay, 'enlarge')
        elif mode == 'normal':
            stride = 1
            mult = 1
            s = x
        else:
            raise Exception("Invalid residual block mode: {}".format(mode))

        h = create_conv2d(h, int(mult * n_filters / 4), 1, 1, weight_decay, 'conv2d_1')
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_2')
        h = tf.nn.relu(h, name='relu_2')
        h = create_conv2d(h, int(mult * n_filters / 4), 3, stride, weight_decay, 'conv2d_2')
        h = tf.layers.batch_normalization(h, training=training_mode, name='bn_3')
        h = tf.nn.relu(h, name='relu_3')
        h = create_conv2d(h, int(mult * n_filters), 1, 1, weight_decay, 'conv2d_3')
        h = h + s

    return h


def create_block_group(input_tensor, block, n_blocks, n_filters, mode, training_mode, weight_decay, name):
    x = input_tensor
    with tf.variable_scope(name):
        for i in range(n_blocks - 1):
            x = block(x, n_filters, 'normal', training_mode, weight_decay, 'block_{}'.format(i+2))
        x = block(x, n_filters, mode, training_mode, weight_decay, 'block_1')

    return x


def create_graph(input_tensor, training_mode, weight_decay, n_classes, block, n_blocks):

    x = input_tensor

    with tf.variable_scope('features'):
        x = create_conv2d(x, 32, 3, 1, weight_decay, 'conv1')
        x = create_block_group(x, block, n_blocks[0], 32, 'enlarge', training_mode, weight_decay, 'group_1')
        x = create_block_group(x, block, n_blocks[1], 64, 'downsample', training_mode, weight_decay, 'group_2')
        x = create_block_group(x, block, n_blocks[2], 128, 'downsample', training_mode, weight_decay, 'group_3')
        x = create_block_group(x, block, n_blocks[3], 256, 'downsample', training_mode, weight_decay, 'group_4')
        x = keras.layers.GlobalAveragePooling2D()(x)

    with tf.variable_scope('classifier'):
        logits = tf.layers.dense(x, n_classes,
                             kernel_initializer=keras.initializers.he_uniform(),
                             kernel_regularizer=keras.regularizers.l2(weight_decay))

    return logits


def create(input_tensor, training_mode, weight_decay, n_classes, depth):

    if depth == 18:
        logits = create_graph(input_tensor, training_mode, weight_decay, n_classes, basic_block, [2,2,2,2])
    elif depth == 34:
        logits = create_graph(input_tensor, training_mode, weight_decay, n_classes, basic_block, [3,4,6,3])
    elif depth == 50:
        logits = create_graph(input_tensor, training_mode, weight_decay, n_classes, bottleneck_block, [3,4,6,3])
    elif depth == 101:
        logits = create_graph(input_tensor, training_mode, weight_decay, n_classes, bottleneck_block, [3,4,23,3])
    elif depth == 152:
        logits = create_graph(input_tensor, training_mode, weight_decay, n_classes, bottleneck_block, [3,8,36,3])
    else:
        raise Exception("Depth must be either 18, 34, 50, 101 or 152. Depth was: {}".format(depth))

    return logits

