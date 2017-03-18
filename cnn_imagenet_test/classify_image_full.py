import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3

from scipy.misc import imread, imresize
import numpy as np
import os
import re
import cPickle as pickle
import glog as log

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('img_file', 'cropped_panda.jpg', 'image to classify')
tf.app.flags.DEFINE_string('ckpt', '../inception_v3.ckpt', 'inception ckpt to start')
tf.app.flags.DEFINE_string('model_dir', '.', 'inception ckpt to start')


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, '../../imagenet/imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, '../../imagenet/imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


if __name__ == '__main__':
  node_lookup = NodeLookup()
  
  image_data = tf.gfile.FastGFile(FLAGS.img_file, 'rb').read()
  image = tf.image.decode_jpeg(image_data, channels=3)
  # im = imread(FLAGS.img_file)
  # im = imresize(im, (299, 299, 3)).astype(np.float32)
  # im = im / 255.
  # im = (im - .5) * 2.

  # im_input = tf.placeholder(tf.float32, [299, 299, 3])
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [299, 299], align_corners=False)
  image = tf.squeeze(image)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  image = tf.expand_dims(image, 0)

  # embed is a Tensor of shape [batch_size, 1000]
  # with tf.variable_scope("InceptionV3"):
  with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
    logits, endpoints = inception_v3(image, num_classes=1001)
  prediction = endpoints['Predictions']

  # use these to get embedding 2048 dim feature
  # with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
    # embed, _ = inception_v3_base(im_input)
  # with tf.variable_scope("InceptionV3"):
  # embed, end_points = my_inception_v3(image, trainable=True, is_training=True)
  #  embed is a Tensor of shape [batch_size, 2048]
  # print(logits.get_shape())

  # all the variables
  
  inception_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
  log.info('printing variables')
  for var in inception_variables:
    print var.name, var.get_shape()

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  # setup initialization
  saver = tf.train.Saver(inception_variables)
  # saver.restore(sess, FLAGS.ckpt)
  saver.restore(sess, '../../inception_v3.ckpt')
  

  ''' data = {}
  for var in inception_variables:
  	data[var.name] = sess.run(var)
  with open('meta2', 'wb') as fo:
    pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)

  # sess.run(init_fn)'''

  logits_np = np.squeeze(sess.run(prediction))

  top_k = logits_np.argsort()[-5:][::-1]
	
  for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = logits_np[node_id]
    print('%s (score = %.5f)' % (human_string, score))

  print 'image'
  print sess.run(image)
  
  # run a variable to see
  for var in inception_variables:
    if var.name == 'InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean:0':
      print sess.run(var)

  '''x =  sess.run(embed)
  print x
  print x.max()
  print x.min()
  # print np.argmax(logits_np)
  # print len(logits_np)
  '''
  sess.close()
