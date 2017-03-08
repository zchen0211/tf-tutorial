import tensorflow as tf
import numpy as np
import glog as log


# problem
batch_size = 10
dim = 3

sample_x_flag = True
sample_w_true_flag = True

tf.set_random_seed(1)

np_transform = np.array([[1., 2., 2.], [0., 1., 2.], [0., 0., 1.]]).astype(np.float32)
np_transform_inv = np.linalg.inv(np_transform)
    
# Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np_transform_inv.astype(np.float32), trainable=True)
Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np.identity(dim).astype(np.float32), trainable=True)

transform = tf.constant(np_transform, dtype=tf.float32)
Wf = tf.get_variable('Wf', shape=[dim, 1], dtype=tf.float32, trainable=True) 

if sample_w_true_flag:
  W_true = tf.placeholder(tf.float32, shape=[dim, 1])
else:
  W_true = tf.get_variable('W_true', shape=[dim, 1], dtype=tf.float32, trainable=False)
    
if sample_x_flag:
  log.info('sampled x')
  x = tf.placeholder(tf.float32, shape=[batch_size, dim])
else:
  x = tf.get_variable('x', shape=[batch_size, dim], dtype=tf.float32, trainable=False)

# x = tf.random_normal([batch_size, dim], 0., 1., dtype = tf.float32)
x_transform = tf.matmul(x, transform)
print 'transform shape:', x_transform.get_shape()
    
y_eps = tf.random_normal([batch_size, 1], 0., 0.1, dtype = tf.float32)
# y_eps = tf.placeholder(tf.float32, [batch_size, 1])
y_true = tf.matmul(x_transform, W_true) + y_eps
    
y_pred = tf.matmul(tf.matmul(x_transform, Ws), Wf)
W_pred = tf.matmul(Ws, Wf)
loss = tf.reduce_sum(tf.square(y_pred - y_true))

for var in tf.trainable_variables():
  print var.name

# optimizer class
optimizer = tf.train.GradientDescentOptimizer(1e-3)

# Phase 1: Compute gradient
# output a list of length |vars|
# each is a tuple (grad_tensor, variable)
# no learning rate introduced in this step
grad_fast = optimizer.compute_gradients(loss, [Wf])
grad_fast, _ = grad_fast[0]
grad_slow = optimizer.compute_gradients(loss, [Ws])
grad_slow, _ = grad_slow[0]

# Phase 2: Apply gradient
fast_op = optimizer.apply_gradients(zip([grad_fast], [Wf]))
slow_op = optimizer.apply_gradients(zip([grad_slow], [Ws]))

log.info('Trainable Variables')
for var in tf.trainable_variables():
  log.info(var.name)

### train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print 'true whitening:', np_transform_inv
print 'true param:', sess.run(W_true)

W_true_np = np.random.normal(size=[dim, 1]).astype(np.float32)
for i in range(500):
  xnp = np.random.normal(size=[batch_size,dim]).astype(np.float32)
  # print 'fast var before update:', sess.run(Wf)
  # log.info('fast_grad')
  # print sess.run(grad_fast, feed_dict={x: xnp})
  sess.run(fast_op, feed_dict={x: xnp, W_true:W_true_np})
  # print 'fast var', sess.run(Wf, feed_dict={x:xnp})
  sess.run(slow_op, feed_dict={x: xnp, W_true:W_true_np})
  # print sess.run(y_eps)
  if i % 10 == 0:
    log.info('step %d, loss: %f' % (i,sess.run(loss, feed_dict={x:xnp, W_true:W_true_np}))) #, feed_dict={y_eps: y_eps_np})
  if (i+1) % 100 == 0:
    print 'slow learner', sess.run(Ws)
    print 'fast learner', sess.run(Wf)
    print 'combined', sess.run(W_pred)
