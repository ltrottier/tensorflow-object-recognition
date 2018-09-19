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

# make a simple model
#n_classes = 10
#net = tf.layers.conv2d(input_tensor, 5, 3, padding='same', use_bias=False)
#net = tf.nn.relu(net)
#net = tf.layers.average_pooling2d(net, 32, 1)
#net = tf.reshape(net, [-1, 5])
#logits = tf.layers.dense(net, n_classes)

# compute training loss
# loss_tensor = tf.losses.sparse_softmax_cross_entropy(target_tensor, logits)

# statistics

## n errors
# stats_n_errors = tf.get_variable(
#     'stats_n_errors',
#     [],
#     initializer=tf.initializers.zeros(),
#     trainable=False,
#     collections=['stats'])
# 
# predicted_target_tensor = tf.argmax(logits, 1)
# error_tensor = tf.not_equal(predicted_target_tensor, tf.reshape(target_tensor, [-1]))
# n_errors_tensor = tf.reduce_sum(tf.cast(error_tensor, tf.float32))
# stats_n_errors_updated = tf.assign_add(stats_n_errors, n_errors_tensor)
# 
# ## n inputs sum
# stats_n_inputs = tf.get_variable(
#     'stats_n_inputs',
#     [],
#     initializer=tf.initializers.zeros(),
#     trainable=False,
#     collections=['stats'])
# stats_n_inputs_updated = tf.assign_add(stats_n_inputs, bs_cur)
# 
# ## error rate
# stats_error_rate_tensor = stats_n_errors / stats_n_inputs
# stats_error_rate_summary_protobuf = tf.summary.scalar('error_rate', stats_error_rate_tensor)
# 
# ## stats update and reset
# stats_variables = [stats_n_errors, stats_n_inputs]
# stats_variables_updated = [stats_n_errors_updated, stats_n_inputs_updated]
# stats_reset_ops = tf.variables_initializer(stats_variables)
# 
# ## stats fetches and summary
# stats_fetches = {
#     'error rate': stats_error_rate_tensor
# }
# stats_summary_protobuf = tf.summary.merge([stats_error_rate_summary_protobuf])
# 
# # optimizer
# loss_minimizer_op = tf.train.AdamOptimizer().minimize(loss_tensor)

#with tf.Session() as sess:
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
#epoch_init = int(sess.run(epoch_tensor)) + 1
#for epoch in range(epoch_init, n_epochs):

    print("training")
    #loss_sum = 0
    sess.run(iterator_initializer_train, feed_dict=iterator_feed_dict_train)
    sess.run(train_begin)
    #sess.run(stats_reset_ops)

    for it in itertools.count(start=0, step=1):
        try:
            #query = [input_tensor, loss_tensor, loss_minimizer_op]
            #input_value, loss_value, _ = sess.run(query)
            #loss_sum = loss_sum + loss_value
            #print("{}, {}, {:.4f}".format(epoch, it, loss_sum / (it + 1)))
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break

    train_summary_str, epoch = sess.run([train_summary, epoch_tensor])
    summary_writer_train.add_summary(train_summary_str, epoch)

    #summary_writer_train.add_summary(*sess.run(train_summary, epoch_tensor))


    print("testing")

    sess.run(iterator_initializer_test, feed_dict=iterator_feed_dict_test)
    #sess.run(stats_reset_ops)
    sess.run(test_begin)

    for it in itertools.count(start=0, step=1):
        try:
            #query = stats_variables_updated
            #sess.run(query)
            sess.run(test_step)
        except tf.errors.OutOfRangeError:
            break

    test_summary_str, epoch = sess.run([test_summary, epoch_tensor])
    summary_writer_test.add_summary(test_summary_str, epoch)

    #fetches = {'variables': stats_variables, 'fetches': stats_fetches}
    #values = sess.run(fetches)
    #print("epoch: {}, stats: {}".format(epoch, values))

    #stats_summary = sess.run(stats_summary_protobuf)
    #summary_writer_test.add_summary(stats_summary, epoch)




# sess.close()
