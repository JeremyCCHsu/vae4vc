import re
import os
import fnmatch
import threading
# import librosa

import numpy as np
import tensorflow as tf

# data_dir = 'TR_SFFT_log_en_SP'
mFea = 513
# speakers = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 
#             'TF1', 'TF2', 'TM1', 'TM2', 'TM3']
# speaker2id = dict(
#     zip(speakers, np.asarray(range(10)).astype(np.float32)))
# sentences = range(100001, 100163)

label_bytes = 4  # I don't know how to read variable-length input
frame_bytes = mFea * 4
record_bytes = label_bytes + frame_bytes


def find_files(directory, pattern='.*\.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        # for filename in fnmatch.filter(filenames, pattern):
        for filename in filenames:
            if re.match(pattern, filename):
                files.append(os.path.join(root, filename))
    # [DEBUG]
    # files = sorted(files)
    # with open('test.txt', 'w') as f:
    #     for filename in files:
    #         f.write('{:s}\n'.format(filename))
    return files

def vc2016TFReader(
    datadir,
    batch_size,
    num_examples_per_epoch,
    num_preprocess_threads=10,
    min_fraction_of_examples_in_queue=0.4,
    pattern='*.bin',
    feature='log-spectrum'):
    min_after_dequeue = int(
        num_examples_per_epoch * min_fraction_of_examples_in_queue)
    capacity = min_after_dequeue + 3 * batch_size

    files = find_files(datadir, pattern=pattern)
    print('Reading {:d} files from {:s}'.format(len(files), datadir))
    # filenames = os.listdir(data_dir)
    # filenames = [
    #     os.path.join(data_dir, f) for f in filenames
    #     if f is not os.path.isdir(f)]

    with tf.variable_scope('input'):
        filename_queue = tf.train.string_input_producer(files)

        # Specify length of record
        reader = tf.FixedLengthRecordReader(record_bytes)

        # Pass 'file_queue' to 'read' method
        key, value = reader.read(filename_queue)
        # [TODO] key should be kept; it's the filename?

        # [ERROR] It gives me 2056 numbers = =
        # floats = tf.decode_raw(value, tf.float32, little_endian=True)
        # value_as_ints = tf.decode_raw(value, tf.float32, little_endian=True)
        # label = value_as_ints[0]
        # [IMPOTANT TIPS]
        #   http://stackoverflow.com/questions/35235697/reading-binary-data-in-float32

        # Decode the result from bytes to float32
        floats = tf.decode_raw(value, tf.float32, little_endian=True)
        # frame = tf.reshape(value_as_floats[1:1+mFea], [1, mFea])
        # label = tf.reshape(floats[0], [1, 1])
        frame = tf.reshape(floats[1: 1 + mFea], [1, mFea])
        # Can't be int, because it'll (y) need to combine with z
        # label_batch = tf.reshape(floats[0], [1, 1])

        if feature == 'spectrum':
            frame = tf.pow(10.0, frame)

        num_labels = 10
        zero = tf.zeros((1, 1), dtype=tf.int64)
        label = tf.reshape(floats[0], [1, 1])
        label = tf.cast(label, tf.int64)
        label = tf.concat(1, [zero, label])
        label = tf.sparse_to_dense(label, [1, num_labels], 1.0)

        # [IMPORTANT] Conversion to onehot (no use for me)
        # http://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
        #
        # [IMPORTANT] The first dimension indicates "how many" samples
        # to read from each file.
        # Leave the first value is OK.
        # In this case, the shape of frame is (513,)
        # and later on `frames, labels = tf.train.shuffle_batch`
        # will give you a batch of [batch_size, 513]
        # but the batch will be read in order.
        # [IMPORTANT] tf.train.shuffle_batch shuffles 'filename'
        # as opposed to samples!

        # batch_size = 2048
        # batch generator
        frames, labels = tf.train.shuffle_batch(
            [frame, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            enqueue_many=True,
            min_after_dequeue=min_after_dequeue)

    return frames, labels



def vc2016TFWholeReader(
        datadir,
        # batch_size=None,
        mFea=514,
        pattern='*.bin',
        feature='log-spectrum',
        output_filename=False):
    '''
    Output: rows of [label, spectrum]
            *need to strip off the label outside the function
    '''
    # mFea = 514
    # datadir = '/home/jrm/proj/vc2016b/SF1-TM3-trn'


    files = find_files(datadir, pattern=pattern)
    # print(files)

    filename_queue = tf.train.string_input_producer(
        files, name='filename_queue')

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    spectrum = tf.decode_raw(
        value,
        tf.float32,
        little_endian=True,
        name='y_x_raw')

    spectrum = tf.reshape(spectrum, [-1, mFea])  
    
    label = spectrum[:, 0]
    spectrum = spectrum[:, 1:]

    if output_filename:
        return label, spectrum, key
    else:
        return label, spectrum


