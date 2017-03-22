import tensorflow as tf
import numpy as np


tmp_y_np = np.ones((1, 10)).astype(np.float32)
tmp_y = tf.get_variable('tmp_y', initializer=tmp_y_np)
tmp_logit = tf.get_variable('tmp_logit', initializer=tmp_y_np)
