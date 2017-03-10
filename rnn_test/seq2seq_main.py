import tensorflow as tf
import numpy as np

if __name__ == '__main__':
  logits_np = np.array([[1., -1.], [-1., 1.], [1., -1.]]).astype(np.float32)
  target_np = np.array([0, 0, 0]).astype(np.int64)

  logits = tf.get_variable('logits', dtype=tf.float32, initializer=logits_np)
  targets = tf.get_variable('targets', dtype=tf.int64, initializer=target_np)
  weights = tf.ones([3], dtype=tf.float32)

  # loss is a tensor of shape [3], need to reduce_mean to serve as a loss
  loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights])

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  print sess.run(loss)
