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
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
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


def my_inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3"):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)
        with tf.variable_scope("logits"):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_inception_model_training,
              scope="dropout")
          net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net, end_points


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
  # prediction = endpoints['Predictions']

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
  
  # softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/softmax:0')

  sess = tf.Session()
  
  # setup initialization
  saver = tf.train.Saver(inception_variables)
  # saver.restore(sess, FLAGS.ckpt)
  saver.restore(sess, '../inception_v3.ckpt')

  sess.run(tf.global_variables_initializer())

  for var in inception_variables:
    if var.name == 'InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean:0':
      print(sess.run(var))
  '''
  data = {}
  for var in inception_variables:
  	data[var.name] = sess.run(var)
  with open('meta2', 'wb') as fo:
    pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)

  # sess.run(init_fn)
  '''

  '''
  logits_np = np.squeeze(sess.run(prediction))

  top_k = logits_np.argsort()[-5:][::-1]
	
  for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = logits_np[node_id]
    print('%s (score = %.5f)' % (human_string, score))
  '''

  '''
  print sess.run(image)

  x =  sess.run(embed)
  print x
  print x.max()
  print x.min()
  # print np.argmax(logits_np)
  # print len(logits_np)
  '''
  sess.close()
