import tensorflow as tf


x = tf.get_variable('x', shape=[3], dtype=tf.float32)
loss = tf.reduce_mean(tf.square(x))
accuracy = 1 - loss

# Study of learning how to draw a curve for a scalar like loss or accuracy
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

lr = 1e-3
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Summary writer writes the graph
summary_writer = tf.summary.FileWriter('.', sess.graph)

for i in range(1000):
  sess.run(train_step)
  if i % 10 == 0:
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, i)

sess.close()
