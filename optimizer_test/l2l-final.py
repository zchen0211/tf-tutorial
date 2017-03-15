import tensorflow as tf
import numpy as np
import glog as log


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch_size')
tf.app.flags.DEFINE_integer('dim', 3, 'problem dimension')
tf.app.flags.DEFINE_integer('sgd_step', 1, 'problem dimension')
tf.app.flags.DEFINE_float('lr_fast', 2e-1, 'lr of fast learner')
tf.app.flags.DEFINE_float('lr_slow', 1e-3, 'lr of slow learner')
tf.app.flags.DEFINE_integer('sample_step', 1000000, 'epochs')
tf.app.flags.DEFINE_integer('visualize_step', 500, 'print inteval')
tf.app.flags.DEFINE_integer('gpu_id', 1, 'number of gpus')

# problem
visualize_step = 500 # 100

def main(_):
  batch_size = FLAGS.batch_size
  sgd_step = FLAGS.sgd_step
  dim = FLAGS.dim
  lr_fast = FLAGS.lr_fast
  lr_slow = FLAGS.lr_slow
  sample_step = FLAGS.sample_step
  visualize_step = FLAGS.visualize_step

  sample_x_flag = True
  sample_w_true_flag = True
  old_update_flag = False  # to enable back-prop for grad(Wf) to update Ws
  print_debug_flag = False

  # set a random seed to make the experiment reproducible
  with tf.device('/gpu:%d' % FLAGS.gpu_id):
    tf.set_random_seed(1)
    np.random.seed(123)

    # generate off-diagonal elements, sample from a N(0, 1)
    np_random = np.random.normal(size=[dim*(dim-1)/2])
    np_transform = np.identity(dim)
    count = 0
    for i in range(dim):
      for j in range(i+1, dim):
        np_transform[i,j] = np_random[count]
        count += 1
    np_transform_inv = np.linalg.inv(np_transform)
    transform = tf.constant(np_transform, dtype=tf.float32)
        
    # Slow learner initialization, starting from eye(dim)
    Ws = tf.get_variable('Ws', dtype=tf.float32, initializer = np.identity(dim).astype(np.float32), trainable=True)

    # Fast learner, re-init to zero for each new task
    Wf = tf.get_variable('Wf', shape=[1, dim], dtype=tf.float32, initializer=tf.constant_initializer(0.), trainable=True) 
    reset_fast_op = tf.variables_initializer([Wf])

    if sample_w_true_flag:
      W_true = tf.placeholder(tf.float32, shape=[1, dim])
    else:
      W_true = tf.get_variable('W_true', shape=[1, dim], dtype=tf.float32, trainable=False)
        
    if sample_x_flag:
      x1 = tf.placeholder(tf.float32, shape=[dim, batch_size])
      x2 = tf.placeholder(tf.float32, shape=[dim, batch_size])
    else:
      x1 = tf.get_variable('x1', shape=[dim, batch_size], dtype=tf.float32, trainable=False)
      x2 = tf.get_variable('x2', shape=[dim, batch_size], dtype=tf.float32, trainable=False)

    # Step 1: fast learner update from x1
    with tf.name_scope('graph1'):
      x_transform1 = tf.matmul(transform, x1)
        
      # y_eps = tf.random_normal([batch_size, 1], 0., 0.0001, dtype = tf.float32)
      with tf.name_scope('ground_truth'):
        y_true = tf.matmul(W_true, x_transform1) # + y_eps

      with tf.name_scope('predict'):
        y_pred = tf.matmul(Wf, tf.matmul(Ws, x_transform1))
      with tf.name_scope('loss_fast'):
        loss = 0.5 * tf.reduce_mean(tf.square(y_pred - y_true))

    # Wf update
    grads_Wf = tf.gradients(loss, [Wf])
    grads_Wf = grads_Wf[0]

    # Step 2: slow learner update from x2
    with tf.name_scope('graph2'):
      Wf_new = - lr_fast * grads_Wf
      # Wf_new = lr_fast * tf.transpose(tf.matmul(tf.matmul(Ws, x_transform1), tf.transpose(y_true)))
      with tf.name_scope('loss_fast_update'):
        loss_old_update = 0.5 * tf.reduce_mean(tf.square(tf.matmul(Wf_new,tf.matmul(Ws,x_transform1)) - y_true))

      x_transform2 = tf.matmul(transform, x2)
      with tf.name_scope('ground_truth'):
        y_true2 = tf.matmul(W_true, x_transform2) # + y_eps
      with tf.name_scope('predict'):
        y_pred2 = tf.matmul(Wf_new, tf.matmul(Ws, x_transform2))
      with tf.name_scope('loss_slow'):
        loss_new = 0.5 * tf.reduce_mean(tf.square(y_pred2 - y_true2))

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
      # optimizer_s = tf.train.MomentumOptimizer(lr_slow, 0.9)
      # Phase 1: Compute gradient
      grad_slow = optimizer_s.compute_gradients(loss_new, [Ws])
      grad_slow, _ = grad_slow[0]

      # Phase 2: Apply gradient
      slow_op = optimizer_s.apply_gradients(zip([grad_slow], [Ws]))

    ### train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('.', sess.graph)

    print 'true whitening:', np_transform_inv

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
          old_loss = sess.run(loss, feed_dict={x1:xnp1, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})
          # fast op
          sess.run(fast_op, feed_dict={x: xnp1, W_true:W_true_np})
          new_loss = sess.run(loss, feed_dict={x1:xnp1, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})

          # slow learner update
          old_loss_slow = sess.run(loss, feed_dict={x1:xnp2, W_true:W_true_np})
          sess.run(slow_op, feed_dict={x:xnp2, W_true:W_true_np})
          new_loss_slow = sess.run(loss, feed_dict={x1:xnp2, W_true:W_true_np})
        else:
          # to monitor loss decrease from 1-sgd fast-learner update
          old_loss = sess.run(loss, feed_dict={x1:xnp1, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})
          new_loss = sess.run(loss_old_update, feed_dict={x1:xnp1, W_true:W_true_np}) #, feed_dict={y_eps: y_eps_np})
          
          # slow learner update
          old_loss_slow = sess.run(loss_new, feed_dict={x1:xnp1, x2:xnp2, W_true:W_true_np})
          sess.run(slow_op, feed_dict={x1: xnp1, x2:xnp2, W_true:W_true_np})
          new_loss_slow = sess.run(loss_new, feed_dict={x1:xnp1, x2:xnp2, W_true:W_true_np})

        if (sample_w+1) % visualize_step == 0:
          if print_debug_flag:
            print 'fast var:'
            if old_update_flag:
              print sess.run(Wf, feed_dict={x1:xnp1})
            else:
              print sess.run(Wf_new, feed_dict={x1:xnp1, W_true:W_true_np})

            print 'grad slow:'
            if old_update_flag:
              print sess.run(grad_slow, feed_dict={x:xnp2, W_true:W_true_np})
            else:
              print sess.run(grad_slow, feed_dict={x:xnp1, x2:xnp2, W_true:W_true_np})

          log.info('fast learner: old / new loss: %f, new loss: %f' % (old_loss, new_loss) )
          log.info('slow learner: old / new loss %f, new loss %f' % (old_loss_slow, new_loss_slow))
          Ws_np = sess.run(Ws)
          print 'Current slow learner with determinant %f: ' % np.linalg.det(Ws_np)
          print Ws_np

          print 'Ws^T * Ws'
          print np.matmul(np.transpose(Ws_np), Ws_np)
          print 'True Sigma inv / eta (theoretical bound):'
          print True_sigma_inv / lr_fast
        
    # print 'In comparison with the slow learner, the true whitening:', np_transform_inv
    sess.close()

if __name__ == '__main__':
  tf.app.run()
