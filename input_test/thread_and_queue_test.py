# Material comes from 
# https://www.tensorflow.org/programmers_guide/threading_and_queues

import tensorflow as tf
import glog as log


q = tf.FIFOQueue(3, 'float')
init = q.enqueue_many(([0., 0., 0.], ))
# the following line is not correct, why?
# init = q.enqueue_many([0., 0., 0.])

x = q.dequeue()
y = x + 1.
q_inc = q.enqueue([y])

sess = tf.Session()

sess.run(init)
sess.run(q_inc)
sess.run(q_inc)
sess.run(q_inc)
sess.run(q_inc)
# log.info('Queue size: %d' % q.size())
print 'Queue size: ', q.size()

# dequeue a tensor, use dequeue_many(n) to return n tensors
z = q.dequeue()
for i in range(3):
  print sess.run(z)
# z_all = q.dequeue_many(3)
# print sess.run(z_all)
