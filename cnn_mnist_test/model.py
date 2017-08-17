#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

MODEL_VARIABLES = '_model_variables_'
VARIABLES_TO_RESTORE = '_variables_to_restore_'
UPDATE_OPS_COLLECTION = '_update_ops_' 

def variable_device(device, name):
    """Fix the variable device to colocate its ops."""
    if callable(device):
        var_name = tf.get_variable_scope().name + '/' + name
        var_def = graph_pb2.NodeDef(name=var_name, op='Variable')
        device = device(var_def)
    if device is None:
        device = ''
    return device

def variable(name, shape=None, dtype=tf.float32, initializer=None,
             regularizer=None, trainable=True, collections=None, device='',
             restore=True):

    collections = list(collections or [])

    # Make sure variables are added to tf.GraphKeys.VARIABLES and MODEL_VARIABLES
    collections += [tf.GraphKeys.VARIABLES, MODEL_VARIABLES]
    # Add to VARIABLES_TO_RESTORE if necessary
    if restore:
        collections.append(VARIABLES_TO_RESTORE)
    # Remove duplicates
    collections = set(collections)
    # Get the device for the variable.
    with tf.device(variable_device(device, name)):
        return tf.get_variable(name, shape=shape, dtype=dtype,
                               initializer=initializer, regularizer=regularizer,
                               trainable=trainable, collections=collections)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")
    # return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform_initializer(minval = low, maxval = high, dtype = tf.float32)


# @scopes.add_arg_scope
def batch_norm2(inputs, epsilon=0.01, decay=0.99, name='bn', train=True):
    shape = inputs.get_shape().as_list()
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    print params_shape
    center = True
    scale = False

    with tf.variable_scope(name) as scope:
        # ema = tf.train.ExponentialMovingAverage(decay=decay)
        if not train:
            scope.reuse_variables()
        beta, gamma = None, None
        if scale:
            gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        if center:
            beta = tf.get_variable("beta", [shape[-1]],
                                   initializer=tf.constant_initializer(0.))
        moving_mean = tf.get_variable('moving_mean', params_shape,
                               initializer=tf.zeros_initializer,
                               trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape,
                                   initializer=tf.ones_initializer,
                                   trainable=False)
        # maintain_averages_op = ema.apply([moving_mean, moving_variance])

        if train:
            inputs_shape = inputs.get_shape()
            axis = list(range(len(inputs_shape) - 1))
            mean, variance = tf.nn.moments(inputs, axis)
            
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
            
            # update mean and variance by adding to an op
            # tf.add_to_collection(UPDATE_OPS_COLLECTION, maintain_averages_op)
        else:
            mean = moving_mean
            variance = moving_variance

    	return tf.nn.batch_norm_with_global_normalization(inputs, mean,
                     variance, beta, gamma, epsilon,
                     scale_after_normalization=True)  # , mean, variance


def BatchNorm(inputT, is_training=True, scope=None):
    if is_training:
    	return batch_norm(inputT, is_training=True, updates_collections=None, scope=scope),
    else:
    	return batch_norm(inputT, is_training=False, updates_collections=None, scope=scope, reuse = True)


class CNN():
	def __init__(self, batch_size=50, image_shape=[28,28,1], c_num=128, label_cnt=10, batch_norm_flag=True):
		self.batch_size = batch_size
		self.batch_norm_flag = batch_norm_flag
		self.c_num = c_num
		self.W_conv1 = tf.get_variable('weight1', [3, 3, 1, c_num],
		                   initializer=xavier_init(c_num, c_num),
		                   trainable=True)
		self.b_conv1 = tf.get_variable('bias1', [c_num],
		                   initializer=tf.constant_initializer(0.1),
		                   trainable=True)
		self.W_conv2 = tf.get_variable('weight2', [3, 3, c_num, c_num],
		                   initializer=xavier_init(c_num, c_num),
		                   trainable=True)
		self.b_conv2 = tf.get_variable('bias2', [c_num],
		                   initializer=tf.constant_initializer(0.1),
		                   trainable=True)
		self.W_conv3 = tf.get_variable('weight3', [3, 3, c_num, c_num],
		                   initializer=xavier_init(c_num, c_num),
		                   trainable=True)
		self.b_conv3 = tf.get_variable('bias3', [c_num],
		                   initializer=tf.constant_initializer(0.1),
		                   trainable=True)
		self.W_conv4 = tf.get_variable('weight4', [1, 1, c_num, c_num],
		                   initializer=xavier_init(c_num, c_num),
		                   trainable=True)
		self.b_conv4 = tf.get_variable('bias4', [c_num],
		                   initializer=tf.constant_initializer(0.1),
		                   trainable=True)
		self.W_fc1 = tf.get_variable('weight5', [128, label_cnt],
		                      initializer=xavier_init(c_num, label_cnt),
		                      trainable=True)
		        # initializer=tf.truncated_normal_initializer(stddev=0.1),
		self.b_fc1 = tf.get_variable('bias5', [label_cnt],
		                      initializer=tf.constant_initializer(0.1),
		                      trainable=True)

	def deep_embed(self, x, phase_train=True, reuse=True):
	    # first conv2d
		with tf.variable_scope('Conv1') as scope1:
			# scope1.reuse_variables()
			h_conv1 = conv2d(x, self.W_conv1) + self.b_conv1
			if self.batch_norm_flag:
				h_bn1 = batch_norm2(h_conv1, name='bn1', train=phase_train)
				'''if phase_train:
					h_bn1 = batch_norm(inputs=h_conv1, updates_collections=None, 
						center=True, scale=False, is_training=phase_train, reuse=False, scope=scope1)
				else:
					h_bn1 = batch_norm(inputs=h_conv1, updates_collections=None, 
						center=True, scale=False, is_training=phase_train, reuse=True, scope=scope1)'''
				h_relu1 = tf.nn.relu(h_bn1)
			else:
				h_relu1 = tf.nn.relu(h_conv1)
			h_pool1 = max_pool_2x2(h_relu1)
		# second conv2d
		with tf.variable_scope('Conv2') as scope2:
			# scope2.reuse_variables()
			h_conv2 = conv2d(h_pool1, self.W_conv2) + self.b_conv2
			if self.batch_norm_flag:
				h_bn2 = batch_norm2(h_conv2, name='bn2', train=phase_train)
				'''if phase_train:
					h_bn2 = batch_norm(inputs=h_conv2, updates_collections=None, 
						center=True, scale=False, is_training=phase_train, reuse=False, scope=scope2)
				else:
					h_bn2 = batch_norm(inputs=h_conv2, updates_collections=None, 
						center=True, scale=False, is_training=phase_train, reuse=True, scope=scope2)'''
				h_relu2 = tf.nn.relu(h_bn2)
			else:
				h_relu2 = tf.nn.relu(h_conv2)
			h_pool2 = max_pool_2x2(h_relu2)
		# third conv2d
		with tf.variable_scope('Conv3') as scope3:
			# scope3.reuse_variables()
			h_conv3 = conv2d(h_pool2, self.W_conv3) + self.b_conv3
			if self.batch_norm_flag:
				h_bn3 = batch_norm2(h_conv3, name='bn3', train=phase_train)
				''' if phase_train:
					h_bn3 = batch_norm(inputs=h_conv3, updates_collections=None,
							center=True, scale=False, is_training=phase_train, reuse=reuse, scope=scope3)
				else:
					h_bn3 = batch_norm(inputs=h_conv3, updates_collections=None,
							center=True, scale=False, is_training=phase_train, scope=scope3)'''
				h_relu3 = tf.nn.relu(h_bn3)
			else:
				h_relu3 = tf.nn.relu(h_conv3)
			h_pool3 = max_pool_2x2(h_relu3)
		# fourth conv2d
		with tf.variable_scope('Conv4') as scope4:
			# scope4.reuse_variables()
			h_conv4 = conv2d(h_pool3, self.W_conv4) + self.b_conv4
			if self.batch_norm_flag:
				h_bn4 = batch_norm2(h_conv4, name='bn4', train=phase_train)
				''' if phase_train:
					h_bn4 = batch_norm(inputs=h_conv4, updates_collections=None, 
						center=True, scale=False, is_training=phase_train, reuse=reuse, scope=scope4)
				else:
					h_bn4 = batch_norm(inputs=h_conv4, updates_collections=None, 
						center=True, scale=False, is_training=phase_train, scope=scope4)'''
				h_relu4 = tf.nn.relu(h_bn4)
			else:
				h_relu4 = tf.nn.relu(h_conv4)
			h_pool4 = tf.reshape(max_pool_2x2(h_relu4), [-1, self.c_num])
		#h_pool4 = tf.reshape(h_pool3, [-1, 2*2*128])
		with tf.variable_scope('FC'):
			y = tf.matmul(h_pool4, self.W_fc1) + self.b_fc1
		return y


	def build_model(self):  # build model to train the deep CNN
		x = tf.placeholder(tf.float32, [None, 28, 28, 1])
		y_ = tf.placeholder(tf.int64, shape=[None])
		y = self.deep_embed(x, phase_train=True)
		cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
		train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), y_), tf.float32))
		return x, y_, cross_entropy_mean, train_accuracy  # loss, accuracy'''


	def test(self):
		x = tf.placeholder(tf.float32, [None, 28, 28, 1])
		y_ = tf.placeholder(tf.int64, shape=[None])
		y = self.deep_embed(x, phase_train=False)
		test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), y_), tf.float32))
		return x, y_, test_accuracy
