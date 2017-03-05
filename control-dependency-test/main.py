import tensorflow as tf


a = tf.get_variable('a', [], dtype=tf.float32)
b = tf.get_variable('b', [], dtype=tf.float32)
c = tf.get_variable('c', [], dtype=tf.float32)

assign_a = a.assign(1.)
assign_b = b.assign(2.)

with tf.get_default_graph().control_dependencies([assign_a, assign_b]):
  assign_c = c.assign(3.)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# will only run assign_a if the following line is called
# sess.run(assign_a)

# call assign_c will triger assign_a and assign_b
sess.run(assign_c)

# However, calling this line will not trigger assign_c
# sess.run(assign_a)
# sess.run(assign_b)

print sess.run(a)
print sess.run(b)
print sess.run(c)
