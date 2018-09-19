import os
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.data import Dataset, Iterator

import dataset
import model


n_epochs = 10

# dataset
dataset_name = 'cifar10'
dataset_dir = None
batch_size = 4
shuffle = False
num_workers = 1
drop_last = True
augment = False
dataset_init = dataset.initialize(
    dataset_name,
    dataset_dir,
    batch_size,
    shuffle,
    num_workers,
    drop_last,
    augment)
input_data_ph = dataset_init[0]
target_data_ph = dataset_init[1]
batch_size_ph = dataset_init[2]
input_tensor = dataset_init[3]
target_tensor = dataset_init[4]
iterator_initializer_train = dataset_init[5]
iterator_initializer_test = dataset_init[6]
iterator_feed_dict_train = dataset_init[7]
iterator_feed_dict_test = dataset_init[8]


# network
network_name = 'dummy'
network_args = [10]
loss_name = 'sparse_softmax_cross_entropy'
stats_train_name = 'error_rate'
stats_test_name = 'error_rate'
optimizer_name = 'adam'
momentum = 0.9
nesterov = True
weight_decay = 1e-4
lr_init = 0.001
lr_decay = 0.1
lr_schedule = [1000]
model_init = model.initialize(
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
    lr_schedule)
epoch_tensor = model_init[0]
train_begin = model_init[1]
train_step = model_init[2]
train_summary = model_init[3]
test_begin = model_init[4]
test_step = model_init[5]
test_summary = model_init[6]


# session
sess = tf.Session()

# summary writer
experiment_folder = 'results/debug'
summary_writer_train = tf.summary.FileWriter(
    os.path.join(experiment_folder, 'train'), sess.graph)
summary_writer_test = tf.summary.FileWriter(
    os.path.join(experiment_folder, 'test'), sess.graph)

# variable initializer
sess.run(tf.global_variables_initializer())

epoch = int(sess.run(epoch_tensor))
while epoch < n_epochs - 1:

    print("training")
    sess.run(iterator_initializer_train, feed_dict=iterator_feed_dict_train)
    sess.run(train_begin)
    for it in itertools.count(start=0, step=1):
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break
    train_summary_str, epoch = sess.run([train_summary, epoch_tensor])
    summary_writer_train.add_summary(train_summary_str, epoch)

    print("testing")
    sess.run(iterator_initializer_test, feed_dict=iterator_feed_dict_test)
    sess.run(test_begin)
    for it in itertools.count(start=0, step=1):
        try:
            sess.run(test_step)
        except tf.errors.OutOfRangeError:
            break
    test_summary_str, epoch = sess.run([test_summary, epoch_tensor])
    summary_writer_test.add_summary(test_summary_str, epoch)


summary_writer_train.close()
summary_writer_test.close()
sess.close()
