import tensorflow as tf
import numpy as np


x1 = tf.get_variable('x1', shape=[])
x2 = tf.get_variable('x2', shape=[])
x3 = tf.get_variable('x3', shape=[])

loss = tf.reduce_sum(tf.square(x1) + tf.square(x2) + tf.square(x3))

lr = 1e-3
opt = tf.train.GradientDescentOptimizer(lr)

# grads is a list of gradients
grads = opt.compute_gradients(loss)

for tmp_grad in grads:
  print type(tmp_grad)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print sess.run(x1)
print sess.run(x2)
print sess.run(x3)

tmp = sess.run(grads)
for i in range(3):
  print tmp[i]

a = tf.get_variable('a', dtype=tf.float32, initializer=np.array([2., 2.]).astype(np.float32))
b = tf.get_variable('b', dtype=tf.float32, initializer=np.array([[3.],[3.]]).astype(np.float32))
c = tf.constant(0., dtype=tf.float32)
# loss = 0.5 * tf.square(a*b - c)
loss = a * b

# grad of a to loss
grads_a = tf.gradients(loss, [a])
grads_b = tf.gradients(loss, [b])
grads_ab = tf.gradients(grads_a[0], b)
sess.run(tf.global_variables_initializer())
print 'a', sess.run(a)
print 'a gradient', sess.run(grads_a[0])
print 'b', sess.run(b)
print 'b gradient', sess.run(grads_b[0])
print 'first to a, then to b', sess.run(grads_ab[0])
