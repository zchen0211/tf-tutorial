import tensorflow as tf
import numpy as np

zero_out_module = tf.load_op_library('zero_out.so')
with tf.Session(''):
  print zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

