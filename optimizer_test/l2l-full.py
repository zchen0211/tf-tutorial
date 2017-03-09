import tensorflow as tf
import numpy as np
import glog as log


# problem
batch_size = 100000
dim = 3
sgd_step = 1
sample_step = 1000
visualize_step = 1

sample_x_flag = True
sample_w_true_flag = True

tf.set_random_seed(1)
np.random.seed(123)

np_transform = np.array([[1., 2., 2.], [0., 1., 2.], [0., 0., 1.]]).astype(np.float32)
np_transform_inv = np.linalg.inv(np_transform)
    
# Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np_transform_inv.astype(np.float32), trainable=True)
Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np.identity(dim).astype(np.float32), trainable=True)

transform = tf.constant(np_transform, dtype=tf.float32)
Wf = tf.get_variable('Wf', shape=[dim, 1], dtype=tf.float32, initializer=tf.constant_initializer(0.), trainable=True) 

reset_fast_op = tf.variables_initializer([Wf])

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
    
y_eps = tf.random_normal([batch_size, 1], 0., 0.0001, dtype = tf.float32)
y_true = tf.matmul(x_transform, W_true) + y_eps
    
y_pred = tf.matmul(tf.matmul(x_transform, Ws), Wf)
W_pred = tf.matmul(Ws, Wf)
loss = 0.5 * tf.reduce_mean(tf.square(y_pred - y_true))

for var in tf.trainable_variables():
  print var.name

# optimizer class
optimizer_f = tf.train.GradientDescentOptimizer(1.)
optimizer_s = tf.train.GradientDescentOptimizer(1e-3)
# optimizer_s = tf.train.AdamOptimizer(1e-4)

# Phase 1: Compute gradient
# output a list of length |vars|
# each is a tuple (grad_tensor, variable)
# no learning rate introduced in this step
grad_fast = optimizer_f.compute_gradients(loss, [Wf])
grad_fast, _ = grad_fast[0]

grad_slow = optimizer_s.compute_gradients(loss, [Ws])
grad_slow, _ = grad_slow[0]

# Phase 2: Apply gradient
fast_op = optimizer_f.apply_gradients(zip([grad_fast], [Wf]))
slow_op = optimizer_s.apply_gradients(zip([grad_slow], [Ws]))

log.info('Trainable Variables')
for var in tf.trainable_variables():
  log.info(var.name)

### train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print 'true whitening:', np_transform_inv
# print 'true param:', sess.run(W_true)

for sample_w in range(sample_step):
  W_true_np = np.random.normal(size=[dim, 1]).astype(np.float32)
  sess.run(reset_fast_op)

  for i in range(sgd_step):
    # fast learner update for one step
    xnp = np.random.normal(size=[batch_size,dim]).astype(np.float32)
    xnp2 = np.random.normal(size=[batch_size,dim]).astype(np.float32)

    raw_loss= sess.run(loss, feed_dict={x:xnp2, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})

    # print 'fast var before update:', sess.run(Wf)
    # log.info('fast_grad')
    # print sess.run(grad_fast, feed_dict={x: xnp, W_true:W_true_np})
    sess.run(fast_op, feed_dict={x: xnp, W_true:W_true_np})
    # print 'fast var', sess.run(Wf, feed_dict={x:xnp})
    
    # slow learner update
    
    sess.run(slow_op, feed_dict={x: xnp2, W_true:W_true_np})
    # print sess.run(y_eps)
    if (sample_w+1) % 10 == 0 and (i+1) % visualize_step == 0:
      log.info('Sample a new problem by sampling from W')
      print W_true_np
      log.info('step %d, loss: %f, %f' % (i+1, raw_loss, sess.run(loss, feed_dict={x:xnp2, W_true:W_true_np})))
      print 'slow learner', sess.run(Ws)
      print 'fast learner', sess.run(Wf)
      print 'combined', sess.run(W_pred)
      Ws_np = sess.run(Ws)
      print 'estimated Sigma', np.matmul(np.transpose(Ws_np), Ws_np)
      print 'True Sigma', np.linalg.inv(np.matmul(np.transpose(np_transform), np_transform))

print 'In comparison with the slow learner, the true whitening:', np_transform_inv
