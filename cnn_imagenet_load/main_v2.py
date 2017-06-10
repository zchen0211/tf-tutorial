import tensorflow as tf
import inception
import functools
import inception_v2

import glog as log
import numpy as np

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', '../../inception_v2.ckpt',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

FLAGS = tf.app.flags.FLAGS

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def get_network_fn(num_classes, weight_decay=0.0, is_training=False):
  # get network
  func = inception.inception_v2
  @functools.wraps(func)
  def network_fn(images):
    arg_scope = inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn

if __name__ == '__main__':
  # get network
  num_classes = 1000
  labels_offset = -1
  network_fn = get_network_fn(num_classes=num_classes-labels_offset)
  
  # get preprocessing
  # inception_preprocessing.preprocess_image(image, height, width, is_training)
  # which in turn is preprocess_for_eval(image, height, width)
  # image_preprocessing_fn = preprocess_for_eval(image, height, width)

  # read image
  image_data = tf.gfile.FastGFile('cropped_panda.jpg', 'rb').read()
  image = tf.image.decode_jpeg(image_data, channels=3)

  # preprocess image
  eval_image_size = inception_v2.inception_v2.default_image_size
  image = preprocess_for_eval(image, eval_image_size, eval_image_size)
  images = tf.reshape(image, [-1, 224, 224, 3])

  # define the model
  logits, _ = network_fn(images)

  # restore variables
  variables_to_restore = slim.get_variables_to_restore()
  path = FLAGS.checkpoint_path
  saver = tf.train.Saver(variables_to_restore)
  sess = tf.Session()
  
  # to visualize the network in tensorboard
  summary_writer = tf.summary.FileWriter('.', sess.graph)

  log.info('Restoring parameters...')
  saver.restore(sess, path)

  # evaluating
  result = sess.run(logits)
  print np.max(result), np.argmax(result)
