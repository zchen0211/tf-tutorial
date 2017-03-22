# Tutorial for devolopers available at https://www.tensorflow.org/programmers_guide/reading_data

import tensorflow as tf


### Step 1: list of file names
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])


with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])

  coord.request_stop()
  coord.join(threads)


# Step 2: Optional filename shuffling

# Step 3: Optional epoch limit

# Step 4: Filename queue

### Step 5: A Reader for the file format
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

### Step 6: A decoder for a record read by the reader
# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])
# Binary files: tf.FixedLengthRecordReader with the tf.decode_raw operation.
# Recommended: TFRecords file containing tf.train.Example protocol buffers
# tf.python_io.TFRecordWriter to convert
# Refer to tensorflow/examples/how_tos/reading_data/convert_to_records.py
# as a good example to convert MNIST.

### Step 7: Optional preprocessing
# Refer to tensorflow_models/tutorials/image/cifar10/cifar10_input.py as a good example


# Step 8: Example queue

