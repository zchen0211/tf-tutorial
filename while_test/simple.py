import tensorflow as tf

i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print sess.run(r)
