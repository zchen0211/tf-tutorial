import tensorflow as tf
import numpy as np


# problem
batch_size = 10
dim = 1 # 3

tf.set_random_seed(1)

# np_transform = np.array([[1., 2., 2.], [0., 1., 2.], [0., 0., 1.]]).astype(np.float32)
np_transform = np.identity(1).astype(np.float32)
# np_transform_inv = np.linalg.inv(np_transform)
    
# Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np_transform_inv.astype(np.float32), trainable=True)
# Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np.identity(dim).astype(np.float32), trainable=True)

transform = tf.constant(np_transform, dtype=tf.float32)
# Wf = tf.get_variable('Wf', shape=[dim, 1], dtype=tf.float32, trainable=True) 

W = tf.get_variable('W', shape=[dim, 1], dtype=tf.float32, trainable=False)

W_true = tf.get_variable('W_true', shape=[dim, 1], dtype=tf.float32, trainable=False)
    
x = tf.get_variable('x', shape=[batch_size, dim], dtype=tf.float32, trainable=False)
# x_transform = tf.matmul(x, transform)
    
# y_eps = tf.random_normal([batch_size], 0., 0.1, dtype = tf.float32)
y_eps = tf.placeholder(tf.float32, [batch_size, 1])
print 'y eps shape: ', y_eps.get_shape()
# y_true = tf.matmul(x_transform, W_true) + y_eps
y_true = tf.matmul(x, W_true) # + y_eps
print 'y true shape: ', y_true.get_shape()
y_noisy = y_true + y_eps
print 'y noisy shape: ', y_noisy.get_shape()
    
# y_pred = tf.matmul(tf.matmul(x_transform, Ws), Wf)
y_pred = tf.matmul(x, W)
    
loss = tf.reduce_sum(tf.square(y_pred - y_noisy))

for var in tf.trainable_variables():
  print var.name

# optimizer class
optimizer = tf.train.GradientDescentOptimizer(1e-4)

# Phase 1: Compute gradient
# output a list of length |vars|
# each is a tuple (grad_tensor, variable)
# no learning rate introduced in this step

# grad_fast = optimizer.compute_gradients(loss, [Wf])
# grad_fast, _ = grad_fast[0]
# grad_slow = optimizer.compute_gradients(loss, [Ws])
# grad_slow, _ = grad_slow[0]
grad_ = optimizer.compute_gradients(loss, [W])
grad_, _ = grad_[0]

# Phase 2: Apply gradient
# fast_op = optimizer.apply_gradients(zip([grad_fast], [Wf]))
# slow_op = optimizer.apply_gradients(zip([grad_slow], [Ws]))
train_op = optimizer.apply_gradients(zip([grad_], [W]))

### train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print 'true param w:', sess.run(W_true)

for i in range(1):
  print 'fast var before update:'
  print sess.run(W)
  y_eps_np = 0.01 * np.random.normal(size=[batch_size, 1]).astype(np.float32)
  print 'x', sess.run(x)
  print 'y', sess.run(y_true)
  print 'noisy y', sess.run(y_noisy, feed_dict={y_eps:y_eps_np})
  print 'fast_grad'
  print sess.run(grad_, feed_dict={y_eps: y_eps_np})
  # sess.run(fast_op, feed_dict={y_eps: y_eps_np})
  sess.run(train_op, feed_dict={y_eps: y_eps_np})
  print 'fast var'
  print sess.run(W)
  # sess.run(slow_op, feed_dict={y_eps: y_eps_np})
  # print sess.run(y_eps)
  # print sess.run(loss, feed_dict={y_eps: y_eps_np})

