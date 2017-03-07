import tensorflow as tf
import numpy as np


x1 = tf.get_variable('x1', shape=[], dtype=tf.float32)
x2 = tf.get_variable('x2', shape=[], dtype=tf.float32)

loss = tf.square(x1) + tf.square(x2)

tvars = tf.trainable_variables()
print tvars

# length=2 list of Tensors
grads = tf.gradients(loss, tvars)

grads, _ = tf.clip_by_global_norm(grads, 1.0)

# optimizer class
optimizer = tf.train.GradientDescentOptimizer(1e-3)

# Phase 1: Compute gradient
# output a list of length |vars|
# each is a tuple (grad_tensor, variable)
# no learning rate introduced in this step
grads_and_vars = optimizer.compute_gradients(loss, tvars)

# Phase 2: Apply gradient
train_op = optimizer.apply_gradients(zip(grads, tvars))

# minimize() combine Phase 1 and Phase 2
train_op2 = optimizer.minimize(loss)

### Evaluate
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print ('x1: %f' % sess.run(x1))
print ('x2: %f' % sess.run(x2))

print sess.run(grads_and_vars)

grads_np = sess.run(grads)
print grads_np

sess.run(train_op)
print ('x1: %f' % sess.run(x1))
print ('x2: %f' % sess.run(x2))

sess.run(train_op2)
print ('x1: %f' % sess.run(x1))
print ('x2: %f' % sess.run(x2))
