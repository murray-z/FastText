# -*- coding: utf-8 -*-

import tensorflow as tf
from fast_text import FastText
from config import config
import time
import os
import datetime
import data_helper
import json


def train(config):
    print('parameters: ')
    print(json.dumps(config, indent=4, ensure_ascii=False))

    # load data
    print('load data .....')
    X, y = data_helper.process_data(config)

    # make vocab
    print('make vocab .....')
    word_to_index, label_to_index = data_helper.generate_vocab(X, y, config)

    # padding data
    print('padding data .....')
    input_x, input_y = data_helper.padding(X, y, config, word_to_index, label_to_index)

    # split data
    print('split data .....')
    x_train, y_train, x_test, y_test, x_dev, y_dev = data_helper.split_data(input_x, input_y, config)

    print('length train: {}'.format(len(x_train)))
    print('length test: {}'.format(len(x_test)))
    print('length dev: {}'.format(len(x_dev)))

    print('training .....')
    with tf.Graph().as_default():
        sess_config = tf.ConfigProto(
            allow_soft_placement=config['allow_soft_placement'],
            log_device_placement=config['log_device_placement']
        )
        with tf.Session(config=sess_config) as sess:
            fast_text = FastText(config)

        # training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(config['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(fast_text.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # output dir for models and summaries
        timestamp = str(int(time.time()))
        outdir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        print('writing to {}'.format(outdir))

        # summary for loss and accuracy
        loss_summary = tf.summary.scalar('loss', fast_text.loss)
        acc_summary = tf.summary.scalar('accuracy', fast_text.accuracy)

        # train summary
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # dev summary
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(outdir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # checkpoint dirctory
        checkpoint_dir = os.path.abspath(os.path.join(outdir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=config['num_checkpoints'])

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
                fast_text.input_x: x_batch,
                fast_text.input_y: y_batch,
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, fast_text.loss, fast_text.accuracy],
                feed_dict=feed_dict
            )

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            feed_dic = {
                fast_text.input_x: x_batch,
                fast_text.input_y: y_batch,
                fast_text.dropout_keep_prob: 1.0
            }

            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, fast_text.loss, fast_text.accuracy],
                feed_dict=feed_dic
            )

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # generate batches
        batches = data_helper.generate_batchs(x_train, y_train, config)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % config['evaluate_every'] == 0:
                print('Evaluation:')
                dev_step(x_dev, y_dev, writer=dev_summary_writer)

            if current_step % config['checkpoint_every'] == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print('save model checkpoint to {}'.format(path))

        # test accuracy
        test_accuracy = sess.run([fast_text.accuracy], feed_dict={
            fast_text.input_x: x_test, fast_text.input_y: y_test, fast_text.dropout_keep_prob: 1.0})
        print('Test dataset accuracy: {}'.format(test_accuracy))


if __name__ == '__main__':
    train(config)



