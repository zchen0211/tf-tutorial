import tensorflow as tf
import numpy as np


embed_np = np.reshape(np.array(range(10)), [10,1])
embed_np = np.tile(embed_np, (1, 3)).astype(np.float32)

with tf.device('/cpu:0'):
  embedding = tf.get_variable('embedding', initializer=embed_np)

ids = tf.constant([[0,2,4],[1,3,5]], dtype=tf.int64)
ids_emb = tf.nn.embedding_lookup(embedding, ids)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print sess.run(ids_emb)
