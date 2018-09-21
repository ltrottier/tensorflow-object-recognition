import os
import numpy as np
import tensorflow as tf


def loop(
        train_begin,
        train_step,
        train_summary,
        train_end,
        test_begin,
        test_step,
        test_summary,
        test_end,
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

    # saver
    saver_filename = os.path.join(experiment_folder, "variables.ckpt")
    saver = tf.train.Saver()

    # initialize variables
    if os.path.exists(os.path.join(experiment_folder, "checkpoint")):
        saver.restore(sess, saver_filename)
    else:
        sess.run(tf.global_variables_initializer())

    # get epoch
    epoch = sess.run(epoch_tensor)

    # loop
    while epoch < n_epochs:

        # train
        sess.run(iterator_initializer_train, feed_dict=iterator_feed_dict_train)
        sess.run(train_begin)
        while True:
            try:
                sess.run(train_step)
            except tf.errors.OutOfRangeError:
                break
        train_summary_str = sess.run(train_summary)
        summary_writer_train.add_summary(train_summary_str, epoch)
        sess.run(train_end)

        # test
        sess.run(iterator_initializer_test, feed_dict=iterator_feed_dict_test)
        sess.run(test_begin)
        while True:
            try:
                sess.run(test_step)
            except tf.errors.OutOfRangeError:
                break
        test_summary_str = sess.run(test_summary)
        summary_writer_test.add_summary(test_summary_str, epoch)
        sess.run(test_end)

        # finalize epoch
        epoch = sess.run(epoch_tensor)
        saver.save(sess, saver_filename)


    summary_writer_train.close()
    summary_writer_test.close()
    sess.close()
