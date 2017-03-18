import tensorflow as tf

'''
var_name = 'InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'
x = tf.get_variable(var_name, shape=[192], initializer=tf.constant_initializer(0.))
print x.name
'''
var_name = 'Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean'
with tf.variable_scope('InceptionV3'):
  x = tf.get_variable(var_name, shape=[192], initializer=tf.constant_initializer(0.))
  print x.name

# incep_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
# incep_vars = [x]
incep_vars = tf.global_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(incep_vars)
saver.restore(sess, '../../inception_v3.ckpt')

print sess.run(x)
