import tensorflow as tf


with tf.variable_scope('test'):
  assert tf.get_variable_scope().name == 'test'
  assert tf.get_variable_scope().reuse == False
  v = tf.get_variable('v', shape=[1])
assert v.name == 'test/v:0'

with tf.variable_scope('test', reuse=True):
  v2 = tf.get_variable('v')

print 'current scope has no name: %s' % tf.get_variable_scope().name

for var in tf.global_variables():
  print var.name
