import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3, inception_v3_arg_scope

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


if __name__ == '__main__':
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
  with slim.arg_scope(inception_v3_arg_scope()):
    # with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
    logits, endpoints = inception_v3(image, num_classes=1001, is_training=False)

  # all the variables
  # all_vars = tf.global_variables()
  # for var in all_vars:
  #   print(var.name)
  inception_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
  var_list = {}
  for var in inception_variables:
    tmp_shape = [int(i) for i in var.get_shape()]
    var_list[var.name] = tmp_shape

  with open('incept_var_dict', 'w') as fo:
    pickle.dump(var_list, fo, protocol=pickle.HIGHEST_PROTOCOL)
  
  log.info('printing variables')
  for var in inception_variables:
    print var.name, var.get_shape()
    # x =  var.get_shape()
    y = [int(i) for i in var.get_shape()]
    # y = [int(x[i]) for i in range(len(x))]
    print y
    # for i in range(len(x)):
    #  print int(x[i])
  
  '''
  sess = tf.Session()
  
  # setup initialization
  saver = tf.train.Saver(inception_variables)
  saver.restore(sess, '../../inception_v3.ckpt')

  sess.run(tf.global_variables_initializer())

  for var in inception_variables:  # all_vars:
    if var.name == 'InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean:0':
      # print(var.name)
      print(sess.run(var))

  sess.close()
  '''