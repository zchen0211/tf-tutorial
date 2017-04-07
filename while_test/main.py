import tensorflow as tf
import glog as log


if __name__ == '__main__':
  i = tf.constant(0)
  # c: condition
  c = lambda i: tf.less(i, 10)
  # b: body
  b = lambda i: tf.add(i,1)
  # [i]: loop_vars
  r = tf.while_loop(c, b, [i])
  # return type: r is a tensor, b/c tf.add(i,1) returns a tensor
  log.info('r type is %s' % type(r))

  print i # now zero
  
  sess = tf.Session()
  # sess.run(tf.global_variables_initializer())
  print sess.run(r)
