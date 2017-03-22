import tensorflow as tf


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

# dequeue a tensor, use dequeue_many(n) to return n tensors
z = q.dequeue()
for i in range(3):
  print sess.run(z)
# z_all = q.dequeue_many(3)
# print sess.run(z_all)
