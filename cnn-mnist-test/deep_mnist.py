import tensorflow as tf
import os
from mnist import input_data
import numpy as np
import numpy.matlib
import math
import glog as log
import cPickle as pickle
import struct
import model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'omniglot',
                           """mnist or omniglot""")
tf.app.flags.DEFINE_string('gpu_id', 0,
                           """which gpu to train on""")
tf.app.flags.DEFINE_string('save_dir', '.',
                           """which gpu to train on""")

if __name__ == "__main__":
    data = input_data.read_data_sets("MNIST_data/", one_hot=False)
    model_save_path = FLAGS.save_dir
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # hyper-parameters
    c_num = 128
    batch_num = 50
    max_step = 25000
    
    test_batch_num = 1000  # to avoid out of memory
    test_cnt = len(data.test.images)
    log.info('training on %d images' % len(data.train.images))

    # with tf.variable_scope('Model') as scope:
    mnist_cnn = model.CNN(batch_norm_flag=True)
    xtr, ytr_, cross_ent_loss, train_acc = mnist_cnn.build_model()
    xte, yte_, test_acc = mnist_cnn.test()
    # scope.reuse_variables()

    lr = 1e-3
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_ent_loss)
    batchnorm_updates = tf.get_collection(model.UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(train_step, batchnorm_updates_op)

    # sess = tf.Session()  # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=10)
        
    sess.run(tf.initialize_all_variables())

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    for var in tf.all_variables():
        summaries.append(tf.histogram_summary(var.op.name, var))
    summary_op = tf.merge_summary(summaries)
    summary_writer = tf.train.SummaryWriter(FLAGS.save_dir, sess.graph)
    # saver.restore(sess, './omniglot_model/omniglot_model-80000')

    for tmp_var in tf.all_variables():
        print tmp_var.name

    # training
    
    for step_i in range(max_step):
        batch = data.train.next_batch(batch_num)

        # train_step.run(session=sess, feed_dict={xtr: batch[0], ytr_: batch[1]})
        _, tr_acc = sess.run([train_op, train_acc], feed_dict={xtr: batch[0], ytr_: batch[1]})

        if (step_i+1) % 100 == 0:
            print("step %d, accuracy %.3f" % (step_i+1, tr_acc))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step_i+1)

        # train_op.run(session=sess, feed_dict={xtr: batch[0], ytr_: batch[1]})

        if (step_i+1) % 500 == 0:
            log.info('testing..')

            te_acc = np.array([0.0])
            for j in range(int(math.ceil(test_cnt/float(test_batch_num)))):
                batch_data = data.test.images[j*test_batch_num:
                                  min(test_cnt,(j+1)*test_batch_num)]
                batch_label = data.test.labels[j*test_batch_num:
                                  min(test_cnt,(j+1)*test_batch_num)]

                te_acc += test_acc.eval(session=sess, feed_dict={xte: batch_data, yte_: batch_label})

            te_acc /= math.ceil(float(test_cnt)/test_batch_num)

            print("test accuracy %.3f" % te_acc)

    sess.close()
