import tensorflow as tf


# Zero operation: all tensors not variables!
zero_a = tf.zeros([2,3], dtype=tf.float32)
zero_b = tf.zeros_like(zero_a)

# One initializer: also a tensor
one_a = tf.ones(shape=[3,3])

# Fill is a tensor
fill_a = tf.fill([2,3], 8.)

# Linspace
lin_a = tf.linspace(0., 10., 5)
range_a = tf.range(0., 10., 6)

# will print nothing, since tensors are not variables
for var in tf.global_variables():
  print var.name

with tf.Session() as sess:
  print sess.run(lin_a)
