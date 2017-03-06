import tensorflow as tf
import glog as log


x = tf.get_variable('x', shape=[], initializer=tf.constant_initializer(1.))
y = tf.get_variable('y', shape=[], initializer=tf.constant_initializer(2.))

assign_op = x.assign(0.)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log.info('x init: %f' % sess.run(x))
sess.run(assign_op)
log.info('x now after assigning: %f' % sess.run(x))

# Re-run the initialization function
log.info('re-initializing...')
var = [x]
reset = tf.variables_initializer(var)
sess.run(reset)
log.info('x now reinitialized: %f' % sess.run(x))

# Partial initialization
init_y = tf.variables_initializer([y], name='init_y')
sess.run(init_y)
log.info('x, y are now %f, %f' % (sess.run(x), sess.run(y)))

# a = tf.Variable(0)
with tf.variable_scope('test'):
  a = tf.get_variable(name='a', shape=[1], initializer=tf.constant_initializer(0.), dtype=tf.float32)
  log.info('type of a is: %s' % str(type(a)))

b = tf.get_variable(name='b', shape=[1], initializer=tf.constant_initializer(0.), dtype=tf.float32)
log.info('type of b is: %s' % str(type(b)))

print sess.run(a)

# operation makes a variable to a tensor
a = a + 1
log.info('type of a now is: %s' % str(type(a)))
print sess.run(a)

# assign value to a tensor? incorrect!
# these following three lines will raise error
# assign_op = tf.assign(a, [2])
# sess.run(assign_op)
# print sess.run(a)

# still output 1, initializer can't re-initialize variables
sess.run(tf.global_variables_initializer())
print sess.run(a)

assign_op = tf.assign(b, [1])
sess.run(assign_op)
print sess.run(b)

log.info('All variables in the project are:')
for var in tf.global_variables():
  print var.name

log.info('Now a is renamed as a operation: %s' % a.name)

log.info('However, we can retrieve the original variable a:')
with tf.variable_scope('test', reuse=True):
  x = tf.get_variable(name='a')
  print sess.run(x)

