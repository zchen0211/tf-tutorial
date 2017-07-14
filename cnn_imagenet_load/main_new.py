import tensorflow as tf
import inception
import functools
import inception_resnet_v2
from scipy.misc import imread, imresize
import os

import glog as log
import numpy as np
import collections
import cPickle as pickle

from parse_imagenet import parse_meta, parse_val

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', '../../inception_resnet_v2_2016_08_30.ckpt',
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


def eval_data_dict(sess, feat, data_fn, save_fn):
  data = np.load(data_fn)
  num = len(data['data'])
  batch_size = 50
  if num % batch_size == 0:
    batches = num / batch_size
  else:
    batches = num/batch_size + 1

  np_feat = np.ndarray([0, 1536]).astype(np.float32)
  for i in range(batches):
    im_np = []
    for j in range(i*batch_size, min((i+1)*batch_size,num)):
      if data['data'][j].shape == (299, 299, 3):
        im_np.append(data['data'][j])
        # im_np = data['data'][i*batch_size : min((i+1)*batch_size,num) ]
    im_np = np.array(im_np)
    im_np = im_np.astype(np.float32) / 255.
    im_np = im_np * 2. - 1.

    new_feat = sess.run(feat, feed_dict = {images: im_np})
    np_feat = np.concatenate((np_feat, new_feat), axis=0)
    print np_feat.shape

  feat_dict = {'feat': np_feat}
  with open(save_fn, 'wb') as fo:
    pickle.dump(feat_dict, fo, protocol=pickle.HIGHEST_PROTOCOL)



def evaluate_full(sess, logits, feat, data_path='/media/DATA/ImageNet/val2012'):
  # load all images from ImageNet validation set
  # run the whole cnn and summarize top-1 error
  val_fn_2_id, val_id_2_fn = parse_val()
  nums_ = []
  correct_ = []

  id_to_feat = {} # id to a 1536-dim feature

  for id_ in val_id_2_fn.keys():
    # step 1: extract images and resize
    ims = []
    for fn in val_id_2_fn[id_]:
      fn = os.path.join(data_path, fn)
      im = imread(fn)
      # preprocess
      im = imresize(im, (eval_image_size, eval_image_size, 3))
      if im.shape == (299, 299, 3):
        im = im.astype(np.float32) / 255.
        im = im * 2. - 1.
        ims.append(im)
      else:
        print im.shape

    num_ = len(ims)
    ims = np.array(ims)

    cnt = 0

    # step 2: extract feature 
    result, np_feat = sess.run([logits, feat], feed_dict = {images: ims})
    id_to_feat[id_] = np_feat
    # prediction id
    top_1 = np.argmax(result, axis=1)
    correct = (top_1==id_+1).sum()
    cnt += correct
    nums_.append(num_)
    correct_.append(correct)
    log.info('class %d: correct: (%d/ %d)' % (id_, correct, num_))
    
    log.info('Accumulative: %d/%d, %f' % (sum(correct_), sum(nums_), 
      float(sum(correct_))/float(sum(nums_)) ) )

  # overall performance on all classes
  log.info('Overall: %d/%d, %f' % (sum(correct_), sum(nums_), 
      float(sum(correct_))/float(sum(nums_)) ) )
  # save features
  # with open('val_feat', 'wb') as fo:
  #   pickle.dump(id_to_feat, fo, protocol=pickle.HIGHEST_PROTOCOL)


def get_network_fn(num_classes, weight_decay=0.0, is_training=False):
  # get network
  func = inception.inception_resnet_v2
  @functools.wraps(func)
  def network_fn(images):
    arg_scope = inception.inception_resnet_v2_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn


def eval_feat(data_path, w, b):
  feat_dict = np.load(data_path)
  b = b.reshape((1, 1000))
  nums_ = []
  correct_ = []
  for id_ in feat_dict.keys():
    tmp_feat = feat_dict[id_]
    n = tmp_feat.shape[0]
    # feat * w + b
    tmp_logit = np.dot(tmp_feat, w)
    tmp_logit = tmp_logit + np.tile(b, (n, 1))
    # accuracy
    top_1 = np.argmax(tmp_logit, axis=1)
    correct = (top_1==id_).sum()
    # put in summary
    nums_.append(n)
    correct_.append(correct)
    log.info('class %d: correct: (%d/ %d)' % (id_, correct, n))

  log.info('Overall: %d/%d, %f' % (sum(correct_), sum(nums_), 
      float(sum(correct_))/float(sum(nums_)) ) )

if __name__ == '__main__':
  # get network
  num_classes = 1000
  labels_offset = -1
  network_fn = get_network_fn(num_classes=num_classes-labels_offset)
  
  # get preprocessing
  eval_image_size = inception_resnet_v2.inception_resnet_v2.default_image_size
  # inception_preprocessing.preprocess_image(image, height, width, is_training)
  # which in turn is preprocess_for_eval(image, height, width)
  # image_preprocessing_fn = preprocess_for_eval(image, height, width)

  # read image
  '''image_data = tf.gfile.FastGFile('cropped_panda.jpg', 'rb').read()
  image = tf.image.decode_jpeg(image_data, channels=3)

  # preprocess image
  image = preprocess_for_eval(image, eval_image_size, eval_image_size)
  images = tf.reshape(image, [-1, eval_image_size, eval_image_size, 3])'''

  im = imread('cropped_panda.jpg')
  im = imresize(im, (eval_image_size, eval_image_size, 3))
  im = im.astype(np.float32) / 255.
  im = im * 2. - 1.

  images = tf.placeholder(tf.float32, [None, eval_image_size, eval_image_size, 3])

  # define the model
  logits, end_points = network_fn(images)
  feat = end_points['PreLogitsFlatten']

  # restore variables
  variables_to_restore = slim.get_variables_to_restore()

  for var in variables_to_restore:
    print(var.name, var.get_shape())
    if var.name == 'InceptionResnetV2/Logits/Logits/weights:0':
      w_var = var
    elif var.name == 'InceptionResnetV2/Logits/Logits/biases:0':
      b_var = var

  path = FLAGS.checkpoint_path
  saver = tf.train.Saver(variables_to_restore)
  sess = tf.Session()
  
  # to visualize the network in tensorboard
  # summary_writer = tf.summary.FileWriter('.', sess.graph)

  log.info('Restoring parameters...')
  saver.restore(sess, path)

  # evaluating
  log.info('evaluating...')
  data_path = '/media/DATA/ImageNet/val2012'
  # evaluate_full(sess, logits, feat)

  w_np, b_np = sess.run([w_var, b_var])
  w_np = w_np[:,1:]
  b_np = b_np[1:]
  data = {'W':w_np, 'b':b_np}
  with open('model-resnet', 'wb') as fo:
    pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)

  # eval_feat('./val_feat', w_np, b_np)

  '''
  data_path = '/media/DATA/ImageNet/Extra'
  data_list = os.listdir(data_path)
  data_list = [item for item in data_list if item[-4:]=='data']
  data_list.sort()
  for fn in data_list:
    fn = os.path.join(data_path, fn)
    save_fn = fn[:-5] + '_feat'
    eval_data_dict(sess, feat, fn, save_fn)
  '''

  # print nums_
  # print correct_


    # for i in range(len(result)):
    #   print np.max(result[i,:]), np.argmax(result[i,:])

  '''result = sess.run(logits, feed_dict = {images: im.reshape((1,299,299,3))})
  print np.max(result), np.argmax(result)

  result = result.reshape(1001)
  top_k = result.argsort()[-5:][::-1]
  id_2_words = parse_meta()
  for id_ in top_k:
    log.info('%s (score %f)' %(id_2_words[id_-1], result[id_]))'''
