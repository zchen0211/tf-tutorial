import tensorflow as tf
import config
import model


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
  config.config()
  print(FLAGS.raw_size)
