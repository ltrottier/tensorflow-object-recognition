import os
import numpy as np
import tensorflow as tf


def loop(
        train_begin,
        train_step,
        train_summary,
        test_begin,
        test_step,
        test_summary,
        iterator_initializer_train,
        iterator_initializer_test,
        iterator_feed_dict_train,
        iterator_feed_dict_test,
        epoch_tensor,
        n_epochs,
        experiment_folder):

    # session
    sess = tf.Session()

    # summary writer
    summary_writer_train = tf.summary.FileWriter(
        os.path.join(experiment_folder, 'train'), sess.graph)
    summary_writer_test = tf.summary.FileWriter(
        os.path.join(experiment_folder, 'test'), sess.graph)

    # variable initializer
    sess.run(tf.global_variables_initializer())

    epoch = sess.run(epoch_tensor)
    while epoch < n_epochs - 1:

        # train
        sess.run(iterator_initializer_train, feed_dict=iterator_feed_dict_train)
        sess.run(train_begin)
        while True:
            try:
                sess.run(train_step)
            except tf.errors.OutOfRangeError:
                break
        train_summary_str, epoch = sess.run([train_summary, epoch_tensor])
        summary_writer_train.add_summary(train_summary_str, epoch)

        # test
        sess.run(iterator_initializer_test, feed_dict=iterator_feed_dict_test)
        sess.run(test_begin)
        while True:
            try:
                sess.run(test_step)
            except tf.errors.OutOfRangeError:
                break
        test_summary_str, epoch = sess.run([test_summary, epoch_tensor])
        summary_writer_test.add_summary(test_summary_str, epoch)


    summary_writer_train.close()
    summary_writer_test.close()
    sess.close()
