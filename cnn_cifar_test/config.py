'''
Define parameters for network training
'''
import tensorflow as tf

def config():
    # weight_file_path = '/home/vis/wangyaming/code_tf/model/pretrained/vgg16_weights.npz'
    # print (weight_file_path)
    
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data_dir', '/media/DATA/cifar-10-batches-py', 'folder containing train file')
    tf.app.flags.DEFINE_string('mean_file', 'batch_cifar.meta', 'file of mean, of size 32 * 32')
    tf.app.flags.DEFINE_integer('crop_size', 28, 'cropped image size')
    tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size')
    tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
    tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
    
    # tf.app.flags.DEFINE_string('test_source',
    #    '/home/vis/wangyaming/data/CUB_200_2011/train.txt', 'test list file')
    '''
    tf.app.flags.DEFINE_integer('max_steps', 20000, 'number of batches to run')
    
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                'weather to log device placement')
    tf.app.flags.DEFINE_boolean('use_fp16', False, 'Train the model using fp16.')
    tf.app.flags.DEFINE_string('weight_file', weight_file_path,
        'weights to load from a pre-trained model.')
    tf.app.flags.DEFINE_string('image_root_dir',
        '/home/vis/wangyaming/data/CUB_200_2011/images_256', 'root directory storing images')
    tf.app.flags.DEFINE_string('mean_file',
        '/home/vis/wangyaming/data/CUB_200_2011/data/birds_256_mean.npy', 'mean image file')
    tf.app.flags.DEFINE_integer('num_gpus', 8, """How many GPUs to use.""")'''

