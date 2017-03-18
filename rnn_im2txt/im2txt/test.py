import tensorflow as tf
import numpy as np


a = [[1,1,1], [1,2,3,4,5]]
a_batch = tf.train.batch_join(a, batch_size=2, dynamic_pad=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(a_batch)
