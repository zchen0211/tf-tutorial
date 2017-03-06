import tensorflow as tf

def config():
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_integer('raw_size', 32, 'raw image size')

