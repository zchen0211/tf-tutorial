import os
import sys

import tensorflow as tf


def simple():
  """Simple problem: f(x) = x^2."""
  y = tf.get_variable("y", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.))

  def build():
    """Builds loss graph."""
    x = tf.get_variable("x", shape=[], dtype=tf.float32, initializer=tf.ones_initializer())
    # return tf.square(x, name="x_squared")
    loss = tf.square(x) + tf.square(y)
    return loss

  return build


