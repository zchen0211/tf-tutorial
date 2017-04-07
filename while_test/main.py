import tensorflow as tf
import glog as log


if __name__ == '__main__':
  # example 1:
  '''# i = tf.constant(0)
  i = 0
  
  # c: condition
  c = lambda i: i<10# tf.less(i, 10)
  
  # b: body
  def body(i):
    return tf.add(i,1)
  b = body # lambda i: tf.add(i,1)
  
  # [i]: loop_vars
  r = tf.while_loop(c, b, [i])
  
  # return type: r is a tensor, b/c tf.add(i,1) returns a tensor
  log.info('r type is %s' % type(r))'''

  # example 2
  unroll_len = 10
  fx_array = tf.TensorArray(tf.int32, size=unroll_len)
  c = lambda i, _: i<unroll_len
  def time_step(i, fx_array):
    fx_array = fx_array.write(i, tf.square(i))
    i = i+1
    return i, fx_array
    
  _, fx_array = tf.while_loop(cond=c, body=time_step, loop_vars=[0, fx_array])
  # print i # now zero
  r = fx_array.stack()
  
  sess = tf.Session()
  # sess.run(tf.global_variables_initializer())
  print sess.run(r)
