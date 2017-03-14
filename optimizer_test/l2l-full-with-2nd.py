import tensorflow as tf
import numpy as np
import glog as log


# problem
batch_size = 10000 # 10000  # 100000
dim = 3
sgd_step = 1
sample_step = 500000  # 1000
visualize_step = 500 # 100

lr_fast = 2e-1
lr_slow = 1e-3

sample_x_flag = True
sample_w_true_flag = True
old_update_flag = False
print_debug_flag = False

tf.set_random_seed(1)
np.random.seed(123)

# np_transform = np.array([[1., 2., 2.], [0., 1., 2.], [0., 0., 1.]]).astype(np.float32)
np_random = np.random.normal(size=[dim*(dim-1)/2])
np_transform = np.identity(dim)
count = 0
for i in range(dim):
  for j in range(i+1, dim):
    np_transform[i,j] = np_random[count]
    count += 1
np_transform_inv = np.linalg.inv(np_transform)
    
# Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np_transform_inv.astype(np.float32), trainable=True)
# (pow(1./lr_fast, 0.3) * 
Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np.identity(dim).astype(np.float32), trainable=True)

transform = tf.constant(np_transform, dtype=tf.float32)
Wf = tf.get_variable('Wf', shape=[1, dim], dtype=tf.float32, initializer=tf.constant_initializer(0.), trainable=True) 

reset_fast_op = tf.variables_initializer([Wf])

if sample_w_true_flag:
  W_true = tf.placeholder(tf.float32, shape=[1, dim])
else:
  W_true = tf.get_variable('W_true', shape=[1, dim], dtype=tf.float32, trainable=False)
    
if sample_x_flag:
  # log.info('sampled x')
  x = tf.placeholder(tf.float32, shape=[dim, batch_size])
  x2 = tf.placeholder(tf.float32, shape=[dim, batch_size])
else:
  x = tf.get_variable('x', shape=[dim, batch_size], dtype=tf.float32, trainable=False)
  x2 = tf.get_variable('x2', shape=[dim, batch_size], dtype=tf.float32, trainable=False)
  # x = tf.random_normal([batch_size, dim], 0., 1., dtype = tf.float32)

with tf.name_scope('graph1'):
  x_transform1 = tf.matmul(transform, x)
    
  # y_eps = tf.random_normal([batch_size, 1], 0., 0.0001, dtype = tf.float32)
  with tf.name_scope('ground_truth'):
    y_true = tf.matmul(W_true, x_transform1) # + y_eps

  with tf.name_scope('predict'):
    y_pred = tf.matmul(Wf, tf.matmul(Ws, x_transform1))
  # W_pred = tf.matmul(Wf, Ws)
  with tf.name_scope('loss_fast'):
    loss = 0.5 * tf.reduce_mean(tf.square(y_pred - y_true))

grads_Wf = tf.gradients(loss, [Wf])
grads_Wf = grads_Wf[0]

with tf.name_scope('graph2'):
  Wf_new = - lr_fast * grads_Wf
  # Wf_new = lr_fast * tf.transpose(tf.matmul(tf.matmul(Ws, x_transform1), tf.transpose(y_true)))
  # print 'Wf_new shape: ', Wf_new.get_shape()
  with tf.name_scope('loss_fast_update'):
    loss_old_update = 0.5 * tf.reduce_mean(tf.square(tf.matmul(Wf_new,tf.matmul(Ws,x_transform1)) - y_true))

  x_transform2 = tf.matmul(transform, x2)
  with tf.name_scope('ground_truth'):
    y_true2 = tf.matmul(W_true, x_transform2) # + y_eps
  with tf.name_scope('predict'):
    y_pred2 = tf.matmul(Wf_new, tf.matmul(Ws, x_transform2))
  with tf.name_scope('loss_slow'):
    loss_new = 0.5 * tf.reduce_mean(tf.square(y_pred2 - y_true2))
  # grads_Ws = tf.gradients(loss_new, [Ws])

for var in tf.trainable_variables():
  print var.name

# optimizer class
if old_update_flag:
  optimizer_f = tf.train.GradientDescentOptimizer(lr_fast)
  optimizer_s = tf.train.GradientDescentOptimizer(lr_slow)

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
else:
  optimizer_s = tf.train.GradientDescentOptimizer(lr_slow)
  # Phase 1: Compute gradient
  grad_slow = optimizer_s.compute_gradients(loss_new, [Ws])
  grad_slow, _ = grad_slow[0]

  # Phase 2: Apply gradient
  slow_op = optimizer_s.apply_gradients(zip([grad_slow], [Ws]))

# log.info('Trainable Variables')
# for var in tf.trainable_variables():
#   log.info(var.name)

### train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('.', sess.graph)

print 'true whitening:', np_transform_inv
# print 'true param:', sess.run(W_true)

True_sigma = np.matmul(np_transform, np.transpose(np_transform))
True_sigma_inv = np.linalg.inv(True_sigma)


for sample_w in range(sample_step):
  W_true_np = np.random.normal(size=[1, dim]).astype(np.float32)
  # print 'W_true', W_true_np
  sess.run(reset_fast_op)

  xnp1 = np.random.normal(size=[dim, batch_size]).astype(np.float32)
  xnp2 = np.random.normal(size=[dim, batch_size]).astype(np.float32)
  
  for i in range(sgd_step):
    if old_update_flag:
      # fast learner update for one step
      old_loss = sess.run(loss, feed_dict={x:xnp1, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})
      # fast op
      sess.run(fast_op, feed_dict={x: xnp1, W_true:W_true_np})
      new_loss = sess.run(loss, feed_dict={x:xnp1, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})

      # slow learner update
      old_loss_slow = sess.run(loss, feed_dict={x:xnp2, W_true:W_true_np})
      sess.run(slow_op, feed_dict={x:xnp2, W_true:W_true_np})
      new_loss_slow = sess.run(loss, feed_dict={x:xnp2, W_true:W_true_np})
    else:
      # if (sample_w+1) % visualize_step == 10:
      #  print ('step %d' % (sample_w+1))
      # print sess.run(Wf_new, feed_dict={x:xnp1, W_true:W_true_np})
      old_loss = sess.run(loss, feed_dict={x:xnp1, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})
      # update Wf_new is contained in the code
      new_loss = sess.run(loss_old_update, feed_dict={x:xnp1, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})
      
      # slow learner update
      old_loss_slow = sess.run(loss_new, feed_dict={x:xnp1, x2:xnp2, W_true:W_true_np})
      sess.run(slow_op, feed_dict={x: xnp1, x2:xnp2, W_true:W_true_np})
      new_loss_slow = sess.run(loss_new, feed_dict={x:xnp1, x2:xnp2, W_true:W_true_np})

    # sess.run(slow_op, feed_dict={x: xnp2, W_true:W_true_np})
    # print sess.run(y_eps)
    if (sample_w+1) % visualize_step == 0:
      # print sess.run(grad_fast, feed_dict={x: xnp, W_true:W_true_np})
      # sess.run(fast_op, feed_dict={x: xnp1, W_true:W_true_np})
      if print_debug_flag:
        print 'fast var:'
        if old_update_flag:
          print sess.run(Wf, feed_dict={x:xnp1})
        else:
          print sess.run(Wf_new, feed_dict={x:xnp1, W_true:W_true_np})

        print 'grad slow:'
        if old_update_flag:
          print sess.run(grad_slow, feed_dict={x:xnp2, W_true:W_true_np})
        else:
          print sess.run(grad_slow, feed_dict={x:xnp1, x2:xnp2, W_true:W_true_np})
      # loss1 = sess.run(loss_new, feed_dict={x:xnp1, x2:xnp2, W_true:W_true_np})
      # print 'slow grad new', sess.run(grads_Ws, feed_dict={x:xnp1, x2:xnp2, W_true:W_true_np})
      # print 'slow grad old', sess.run(grad_slow_old, feed_dict={})
      # sess.run(slow_op, feed_dict={x: xnp1, x2:xnp2, W_true:W_true_np})
      # loss2 = sess.run(loss_new, feed_dict={x:xnp1, x2:xnp2, W_true:W_true_np})
      # print 'Wf', sess.run(Wf)
      #print 'Wf new: ', sess.run(Wf_new, feed_dict={x:xnp1, W_true:W_true_np})

      log.info('fast learner: old / new loss: %f, new loss: %f' % (old_loss, new_loss) )
      log.info('slow learner: old / new loss %f, new loss %f' % (old_loss_slow, new_loss_slow))
      # log.info('Sample a new problem by sampling from W')
      # print W_true_np
      # log.info('step %d, loss: %f, %f' % (i+1, raw_loss, sess.run(loss, feed_dict={x:xnp2, W_true:W_true_np})))
      # print 'slow learner', sess.run(Ws)
      # print 'fast learner', sess.run(Wf)
      # print 'combined', sess.run(W_pred)
      Ws_np = sess.run(Ws)
      print 'Current slow learner with determinant %f: ' % np.linalg.det(Ws_np)
      print Ws_np

      print 'Ws^T * Ws', np.matmul(np.transpose(Ws_np), Ws_np)
      print 'True Sigma inv', True_sigma_inv
    
# print 'In comparison with the slow learner, the true whitening:', np_transform_inv
sess.close()

