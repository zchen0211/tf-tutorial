import tensorflow as tf

import numpy as np
import glog as log


if __name__ == '__main__':

  # First, we need various cells as basic building blocks
  cell_size = 10
  cell_lstm = tf.contrib.rnn.BasicLSTMCell(cell_size)
  cell_rnn = tf.contrib.rnn.BasicRNNCell(cell_size)

  # build an RNN by hand
  batch_size = 3
  num_step = 5
  input_np_ = np.array([range(0,num_step), range(1,num_step+1), range(2,num_step+2)])
  input_np_ = input_np_.reshape([batch_size, num_step, 1]).astype(np.float32)
  print input_np_.shape
  input_ = tf.placeholder(tf.float32, shape=[batch_size, num_step,1])
  init_state = cell_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
  state = init_state
  outputs = []
  with tf.variable_scope("RNN"):
    for time_step in range(num_step):
      if time_step > 0:
        tf.get_variable_scope().reuse_variables()
      (cell_output, state) = cell_lstm(input_[:,time_step,:], state)
      outputs.append(cell_output)
  # Now outputs is a length(10) list
  # each item is an ouput tensor of shape [3, 10], batch_size * cell_size

  # rearrange output to compute loss with ground-truth labels
  # first concatenate to size (3, 100) to maintain batch_size, then resize to (30, 10)
  output = tf.reshape(tf.concat(outputs,1), [-1, cell_size])

  # See the graph
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  # summary_writer = tf.summary.FileWriter('.', sess.graph)

  outputs_np = sess.run(outputs, feed_dict={input_: input_np_})
  # print outputs_np
  print outputs_np[0].shape

  # Following lines will print nothing if only cell is declared,
  # since it is not created during declaration
  for var in tf.global_variables():
    log.info('variable: %s' % var.name)
    print var.get_shape()


