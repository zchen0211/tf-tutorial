from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import glog as log

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import config
import model_multi_gpu
import input_data


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', './multi_gpu_model_save', 'directory saving log and models')
tf.app.flags.DEFINE_integer('raw_size', 32, 'raw image size')
tf.app.flags.DEFINE_integer('num_classes', 10, 'number of classes')
tf.app.flags.DEFINE_integer('num_gpus', 2, 'number of gpus')

def tower_loss(scope, custom_runner, first_time_flag=True):
  # only for training
  with tf.device('/cpu:0'):
    images, labels = custom_runner.get_inputs()

  # Build inference Graph.
  logits = model_multi_gpu.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = model_multi_gpu.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  # loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % model_multi_gpu.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    custom_runner = input_data.CustomRunner(train=True)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (model_multi_gpu.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * model_multi_gpu.NUM_EPOCHS_PER_DECAY / FLAGS.num_gpus)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(model_multi_gpu.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    model_multi_gpu.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    # if model_tutorial.OPTIMIZER == 'sgd':
    opt = tf.train.GradientDescentOptimizer(lr)
    # elif model_tutorial.OPTIMIZER == 'mom':
    # opt = tf.train.MomentumOptimizer(lr, model_tutorial.MOMENTUM)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      log.info('Setting up on GPU %d' % i)
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (model_multi_gpu.TOWER_NAME, i)) as scope:
          # Calculate the loss for one tower of the model. This function
          # constructs the entire model but shares the variables across
          # all towers.
          loss = tower_loss(scope, custom_runner, i==0)
          
          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()
          
          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
          
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

      
      for var in tf.global_variables():
        log.info('variable: %s' % var.name)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    
    # clip gradient by value if required
    # grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
    # # set 10x grads (lr) for w and b in the last layer
    # grads[-1] = (grads[-1][0] * 10, grads[-1][1])
    # grads[-2] = (grads[-2][0] * 10, grads[-2][1])
    
    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #    model_multi_gpu.MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    # train_op = tf.group(apply_gradient_op, variables_averages_op)
    train_op = apply_gradient_op

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    
    #queue_size = custom_runner.get_queue_size()
    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.


    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement,
        intra_op_parallelism_threads=16,
        inter_op_parallelism_threads=16))

    sess.run(init)

    # load weights from a pre-trained model.
    # model.load_weights(sess)

    # Start the queue runners.
    #tf.train.start_queue_runners(sess=sess)
    # Start our custom queue runner's threads
    #custom_runner0.start_threads(sess)
    #custom_runner1.start_threads(sess)
    #custom_runner2.start_threads(sess)
    #custom_runner3.start_threads(sess)
    custom_runner.start_threads(sess)
    
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    
    print ('max steps = %d, decay_steps = %d' % (FLAGS.max_steps, decay_steps))

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      #current_queue_size = sess.run(queue_size)
      #print("Current Queue Size is: ", current_queue_size)
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f, lr = %f (%.1f examples/sec; %.3f sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, sess.run(lr), examples_per_sec, sec_per_batch))

      if step % 100 == 0:      
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)        
  train()


if __name__ == '__main__':
  config.config()
  tf.app.run()
