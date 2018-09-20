import os
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.data import Dataset, Iterator


def load_cifar10_dataset(
        dataset_dir,
        batch_size,
        shuffle,
        num_workers,
        drop_last,
        augment):


    # load data
    (x_tr, y_tr), (x_tst, y_tst) = keras.datasets.cifar10.load_data()
    x_tr = (x_tr.astype('float32') / 255.0)
    x_tst = (x_tst.astype('float32') / 255.0)

    n_obs = 1000
    x_tr = x_tr[:n_obs]
    y_tr = y_tr[:n_obs]
    x_tst = x_tst[:n_obs]
    y_tst = y_tst[:n_obs]

    # shape
    input_shape = list(x_tr.shape)
    input_shape[0] = None
    target_shape = list(y_tr.shape)
    target_shape[0] = None

    # create placeholders for data
    input_data_ph = tf.placeholder(tf.float32, shape=input_shape, name='input_data')
    target_data_ph = tf.placeholder(tf.int64, shape=target_shape, name='target_data')

    # create placeholder for batch size
    batch_size_ph = tf.placeholder(tf.int64, name='batch_size')

    # create the train Dataset object
    dataset_train = Dataset.from_tensor_slices((input_data_ph, target_data_ph))
    if drop_last:
        dataset_train = dataset_train.apply(tf.contrib.data.batch_and_drop_remainder(batch_size_ph))
    else:
        dataset_train = dataset_train.batch(batch_size_ph)
    if augment:
        pass
    if shuffle:
        #dataset_train = dataset_train.shuffle()
        pass
    dataset_train = dataset_train.prefetch(num_workers * batch_size)

    # create the test Dataset object
    dataset_test = Dataset.from_tensor_slices((input_data_ph, target_data_ph))
    dataset_test = dataset_test.batch(batch_size_ph)
    dataset_test = dataset_test.prefetch(num_workers * batch_size)

    # create the reinitializable iterators
    dataset_output_types = dataset_train.output_types
    dataset_output_shapes = dataset_train.output_shapes
    iterator = Iterator.from_structure(dataset_output_types, dataset_output_shapes)
    iterator_initializer_train = iterator.make_initializer(dataset_train)
    iterator_initializer_test = iterator.make_initializer(dataset_test)

    # create the feed_dict for the iterators
    iterator_feed_dict_train = {input_data_ph: x_tr, target_data_ph: y_tr, batch_size_ph: batch_size}
    iterator_feed_dict_test = {input_data_ph: x_tst, target_data_ph: y_tst, batch_size_ph: batch_size}

    # input and target tensors
    input_tensor, target_tensor = iterator.get_next(name='sample')

    # dataset
    dataset_init = [
        input_data_ph,
        target_data_ph,
        batch_size_ph,
        input_tensor,
        target_tensor,
        iterator_initializer_train,
        iterator_initializer_test,
        iterator_feed_dict_train,
        iterator_feed_dict_test,
    ]

    return dataset_init

def initialize(
        dataset_name,
        dataset_dir,
        batch_size,
        shuffle,
        num_workers,
        drop_last,
        augment):

    with tf.variable_scope('dataset'):
        if dataset_name == 'cifar10':
            dataset_init = load_cifar10_dataset(
                dataset_dir, batch_size, shuffle, num_workers, drop_last, augment)
        elif dataset_name == 'cifar100':
            dataset_init = load_cifar100_dataset(
                dataset_dir, batch_size, shuffle, num_workers, drop_last, augment)
        elif dataset_name == 'svhn':
            dataset_init = load_svhn_dataset(
                dataset_dir, batch_size, shuffle, num_workers, drop_last, augment)
        else:
            raise Exception("Invalid dataset type: {}".format(dataset_type))

    return dataset_init

