# single GPU training of Cifar

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import glog as log
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import config
import model_tutorial
import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('raw_size', 32, 'raw image size')
tf.app.flags.DEFINE_integer('num_classes', 10, 'number of classes')
tf.app.flags.DEFINE_string('log_dir', './single_gpu_model_save', 'directory saving log and models')


def train():
    """Train for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Obtain a batch of images and labels (on CPU)
        with tf.device("/cpu:0"):
            custom_runner = input_data.CustomRunner(train=True)
            images, labels = custom_runner.get_inputs()

            custom_runner_test = input_data.CustomRunner(train=False)
            images_test, labels_test = custom_runner_test.get_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model_tutorial.inference(images, train=True)
        logits_test = model_tutorial.inference(images_test, train=False)

        # Calculate loss.
        loss = model_tutorial.loss(logits, labels)

        # Accuracy
        accuracy = model_tutorial.accuracy(logits, labels)
        accuracy_test = model_tutorial.accuracy(logits_test, labels_test)

        tf.summary.scalar('training loss', loss)
        tf.summary.scalar('training accuracy', accuracy)
        tf.summary.scalar('testing accuracy', accuracy_test)
        summary_op = tf.summary.merge_all()

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model_tutorial.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Load weights from a pre-trained model.
        # model.load_weights(sess)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        # Start our custom queue runner's threads
        custom_runner.start_threads(sess)
        custom_runner_test.start_threads(sess)

        summary_writer = tf.summary.FileWriter('.', sess.graph)

        # Try train data-provider
        
        '''log.info('testing...')
        images_batch, labels_batch = sess.run([images_test, labels_test])
        print(images_batch.shape)
        print(labels_batch)

        log.info('training...')
        images_batch, labels_batch = sess.run([images, labels])
        print(images_batch.shape)
        print(labels_batch)'''
        

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            images_batch, labels_batch = sess.run([images, labels])    

            # print("Batched Image")
            # print(images_batch[0, 0:10, 0:10, 0])
            # print(labels_batch)

            # Run one step of the model.
            _, batch_accuracy, loss_value = sess.run([train_op, accuracy, loss])
                                     
            # print("train_op done")
            duration = time.time() - start_time
           
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if (step+1) % 100 == 0:  # print training statistics
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                log.info('step: %d, loss: %f, accuracy: %f' % (step+1, loss_value, batch_accuracy))
                # format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                #              'sec/batch)')
                # print (format_str % (datetime.now(), step, loss_value,
                #                     examples_per_sec, sec_per_batch))

            if step % 1000 == 0:  # print testing statistics
                log.info('testing...')
                te_accuracy = 0.
                for i in range(100):
                    images_batch_test, labels_batch_test = sess.run([images_test, labels_test])
                    batch_acc = sess.run(accuracy_test)
                    te_accuracy += batch_acc
                te_accuracy = te_accuracy / 100.
                log.info('Testing Accuracy: %f' % te_accuracy)

                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            
            # Save the model checkpoint periodically.
            # if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            #  checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
            #  saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)        
    train()


if __name__ == '__main__':
    config.config()
    tf.app.run()
