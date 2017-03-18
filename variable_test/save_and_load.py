import tensorflow as tf

import numpy as np
import glog as log
import os
import shutil

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string('task', 'save', 'save or load?')


def save_model(sess, saver, fname):
  all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  log.info('Variables to save')
  assign_op_all = []
  for var in all_var:
    tmp_size = var.get_shape()[0]
    print var.name
    print var
    assign_op = var.assign(np.array(range(tmp_size)).astype(np.float32))
    assign_op_all.append(assign_op)

  sess.run(assign_op_all)
  saver.save(sess, fname)


def load_model(sess, saver, fname):
  saver.restore(sess, fname)


if __name__ == '__main__':
  x = tf.get_variable('x', shape=[100], initializer=tf.constant_initializer(1.))
  y = tf.get_variable('y', shape=[1000], initializer=tf.constant_initializer(1.))
  z = tf.get_variable('z', shape=[10000], initializer=tf.constant_initializer(1.))
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  # every var in the saver list should be saved before, otherwise 
  var_list = [x]  # tf.global_variables()
  saver = tf.train.Saver(var_list)

  tmp_path = 'model/'
  fname = 'test.ckpt'

  if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)

  # variable value after initialization
  for var in tf.global_variables():
    log.info('variable value: %s', var.name)
    print sess.run(var)
  
  # save task
  if FLAGS.task == 'save':
    save_model(sess, saver, os.path.join(tmp_path,fname))
    for var in tf.global_variables():
      log.info('variable value: %s', var.name)
      print sess.run(var)
  elif FLAGS.task == 'load':
    # load and see
    load_model(sess, saver, os.path.join(tmp_path, fname))
    for var in tf.global_variables():
      log.info('variable value: %s', var.name)
      print sess.run(var)
    shutil.rmtree(tmp_path)
  else:
    raise 'Unknown task'
