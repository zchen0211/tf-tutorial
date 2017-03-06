import tensorflow as tf
import collections
import mock
import glog as log


def _make_with_custom_variables(func, variables):
  variables = collections.deque(variables)

  def custom_getter(getter, name, **kwargs):
    if kwargs["trainable"]:
      return variables.popleft()
    else:
      kwargs["reuse"] = True
      return getter(name, **kwargs)

  return _wrap_variable_creation(func, custom_getter)


def _wrap_variable_creation(func, custom_getter):
  """Provides a custom getter for all variable creations."""
  original_get_variable = tf.get_variable
  def custom_get_variable(*args, **kwargs):
    if hasattr(kwargs, "custom_getter"):
      raise AttributeError("Custom getters are not supported for optimizee variables.")
    return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

  # Mock the get_variable method.
  with mock.patch("tensorflow.get_variable", custom_get_variable):
    return func()


def _get_variables(func):
  variables = []
  constants = []

  def custom_getter(getter, name, **kwargs):
    trainable = kwargs["trainable"]
    kwargs["trainable"] = False
    variable = getter(name, **kwargs)
    if trainable:
      variables.append(variable)
    else:
      constants.append(variable)
    return variable

  with tf.name_scope("unused_graph"):
    _wrap_variable_creation(func, custom_getter)

  return variables, constants


def meta_minimize(make_loss):
  meta_loss(make_loss)


def meta_loss(make_loss):
  x, constants = _get_variables(make_loss)

  print("Optimizee variables")
  print([op.name for op in x])
  print("Problem variables")
  print([op.name for op in constants])

  fx = _make_with_custom_variables(make_loss, x)
  log.info(type(fx))
  print fx is None
  
  fx_array = tf.TensorArray(tf.float32, 1, clear_after_read=False)
  fx_array = fx_array.write(0, fx)
  loss = tf.reduce_sum(fx_array.stack(), name="loss")


# problem = simple()
# meta_minimize(problem)
# log.info(type(fx))
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print sess.run(loss)
