import tensorflow as tf

import numpy as np
import glog as log

# First, we need various cells as basic building blocks
cell_size = 2
cell_lstm = tf.contrib.rnn.BasicLSTMCell(cell_size)
cell_rnn = tf.contrib.rnn.BasicRNNCell(cell_size)

# build an RNN by hand
batch_size = 3
num_step = 5
input_np_ = np.array([range(0,num_step), range(1,num_step+1), range(2,num_step+2)])
input_np_ = input_np_.reshape([batch_size, num_step, 1]).astype(np.float32)
print input_np_.shape
input_ = tf.placeholder(tf.float32, shape=[batch_size, num_step,1])

# input_[batch_size, T, cell_size] to [T, batch_size, cell_size] first
# then decompose into length=T list, each of shape [batch_size, cell_size] 
input_unpack = tf.unstack(tf.transpose(input_, perm=[1,0,2]))
init_state = cell_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

# outputs_rnn: a list of length = T, each of shape [batch_size, cell_size]
# outputs_rnn will have all 0s for unvalid sequence length
# state is a correct tuple computed up to valid length
# state[0]: final memory cell [batch_size, cell_size] c
# state[1]: final output [batch_size, cell_size] h

outputs_rnn, state = tf.contrib.rnn.static_rnn(cell_lstm, input_unpack, initial_state=init_state, 
                               sequence_length=[3,4,3])
log.info('state: %s' % type(state))
# loss should be computed with mask like: loss * mask_flag
# or with tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [target], [weight])

# rearrange output to compute loss with ground-truth labels
# first concatenate to size (batch_size, T*cell_size) to maintain batch_size, then resize to (batch_size*T, cell_size)
output = tf.reshape(tf.concat(outputs_rnn,1), [-1, cell_size])
log.info('output size:')
print output.get_shape()

# See the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# summary_writer = tf.summary.FileWriter('.', sess.graph)'''

outputs_np, state_np = sess.run([output, state], feed_dict={input_: input_np_})
log.info('output of rnn')
print outputs_np
# print outputs_np[0].shape
log.info('state:')
print 'c:', state_np[0]
print 'h:', state_np[1]

# Following lines will print nothing if only cell is declared,
# since it is not created during declaration
for var in tf.global_variables():
  log.info('variable: %s' % var.name)
  print var.get_shape()



