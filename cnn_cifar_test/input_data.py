from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import collections
from six.moves import xrange  # pylint: disable=redefined-builtin
# from PIL import Image
import threading
import glog as log

import tensorflow as tf
from tensorflow.python.framework import dtypes
from matplotlib import pylab as plt

FLAGS = tf.app.flags.FLAGS

#MEAN_PIXEL = [123.68, 116.779, 103.939] # mean pixel in RGB

def data_iterator(train=True):
    """ A simple data iterator """
    idx = 0
    # obtain image list with labels
    array, label = parse_data_source(FLAGS.data_dir, train)
    #    FLAGS.image_root_dir)
    
    img_num = len(label)

    while True:
        for idx in range(img_num):
            tmp_image = array[idx]
            tmp_label = label[idx]

            yield tmp_image, tmp_label


class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, train=True):
        self.train = train # training mode or not

        self.dataX = tf.placeholder(dtype=tf.float32, shape=[FLAGS.raw_size, FLAGS.raw_size, 3])
        self.dataY = tf.placeholder(dtype=tf.int64, shape=[])

        # get the mean. 
        mean_ = np.load(os.path.join(FLAGS.data_dir, FLAGS.mean_file))
        mean_ = mean_['data_mean'].astype(np.float32)
        self.mean_dataX = tf.constant(mean_, dtype=tf.float32)

        # mean subtraction
        self.mean_sub_image = self.dataX - self.mean_dataX

        # The actual queue of data. The queue contains a vector for an image and a scalar label.
        if self.train:
            self.queue = tf.RandomShuffleQueue(shapes=[[FLAGS.crop_size, FLAGS.crop_size, 3], []],
                                               dtypes=[tf.float32, tf.int64], capacity=2000, min_after_dequeue=1000)
            # random crop
            self.distorted_image = tf.random_crop(self.mean_sub_image, [FLAGS.crop_size, FLAGS.crop_size, 3])
            # random flip
            self.distorted_image = tf.image.random_flip_left_right(self.distorted_image)
            # random brightness, saturation and contrast
            self.distorted_image = tf.image.random_brightness(self.distorted_image, max_delta=63. / 255.)
            self.distorted_image = tf.image.random_saturation(self.distorted_image, lower=0.5, upper=1.5)
            self.distorted_image = tf.image.random_contrast(self.distorted_image, lower=0.2, upper=1.8)
        else:
            self.queue = tf.FIFOQueue(shapes=[[FLAGS.crop_size, FLAGS.crop_size, 3], []],
                                               dtypes=[tf.float32, tf.int64], capacity=20000)
            # center crop
            self.distorted_image = tf.image.resize_image_with_crop_or_pad(self.mean_sub_image, FLAGS.crop_size, FLAGS.crop_size)
            # tf.image.central_crop(image, central_fraction)
        
        # enqueue
        self.enqueue_op = self.queue.enqueue([self.distorted_image, self.dataY])
        #self.enqueue_op = self.queue.enqueue([self.dataX, self.dataY])


    def get_inputs(self):
        """
        Return tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(FLAGS.batch_size)
        return images_batch, labels_batch


    def get_queue_size(self):
        return self.queue.size()


    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY in data_iterator(self.train):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY})
            # self.mean_dataX:mean_image})


    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(0, n_threads):
            t = threading.Thread(target=self.thread_main,
                                 args=(sess, ))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


def parse_data_source(source_dir, train=True):
    fn = os.listdir(source_dir)
    fn = [item for item in fn if item[:5] == 'data_']
    
    print(fn)

    # training
    if train:
        array = np.ndarray([0, 32, 32, 3]).astype(np.float32)
        label = []
        fn = fn[:-1]
        for tmp_fn in fn:
            full_fn = os.path.join(source_dir, tmp_fn)
            data = np.load(full_fn)

            tmp_im_array = data['data']
            tmp_im_array = tmp_im_array.reshape((10000, 3, 32, 32)).swapaxes(1,3).swapaxes(1,2)
            tmp_im_array = tmp_im_array.astype(np.float32) / 255.
            array = np.concatenate((array, tmp_im_array), axis=0)

            label = label + data['labels']

        log.info(array.shape)
        log.info(len(label))
    else:
    # testing
        array = np.ndarray([0, 32, 32, 3]).astype(np.float32)
        label = []
        fn = [fn[-1]]
        log.info('testing %s' % fn)
        for tmp_fn in fn:
            full_fn = os.path.join(source_dir, tmp_fn)
            data = np.load(full_fn)

            tmp_im_array = data['data']
            tmp_im_array = tmp_im_array.reshape((10000, 3, 32, 32)).swapaxes(1,3).swapaxes(1,2)
            tmp_im_array = tmp_im_array.astype(np.float32) / 255.
            array = np.concatenate((array, tmp_im_array), axis=0)

            label = label + data['labels']

        log.info(array.shape)
        log.info(len(label))

    return array, label
    '''with open(source_file) as f:
        lines = f.readlines()
    num_samples = len(lines)
    img_list = []
    labels = np.zeros((num_samples, ), dtype=np.int32)
    for i, line in enumerate(lines):
        line = line.strip()
        segs = line.split(' ')
        img_list.append(os.path.join(image_root_dir, segs[0]))
        labels[i] = int(segs[1])'''
    # return (np.array(img_list), labels)


if __name__ == '__main__':
    tr_array, tr_label = parse_data_source('/media/DATA/cifar-10-batches-py')
    te_array, te_label = parse_data_source('/media/DATA/cifar-10-batches-py', train=False)
    '''plt.imshow(tr_array[30])
    plt.show()
    plt.imshow(te_array[25])
    plt.show()'''