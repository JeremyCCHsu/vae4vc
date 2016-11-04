import os
import re
import sys
import time
import json

import pdb
import numpy as np
import tensorflow as tf
# [TODO]
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import loadmat
from datetime import datetime
from vae.model import VAE2
from iohandler.spectral_reader import vc2016TFWholeReader
from util.spectral_processing import MelCepstralProcessing

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('source', 'SF1', 'list of source speakers')
# tf.app.flags.DEFINE_string('target', 'TM3', 'list of target speakers')
# tf.app.flags.DEFINE_string('{:s}-{:s}-trn', '', 'data dir')
tf.app.flags.DEFINE_string(
    'datadir', '/home/jrm/proj/vc2016b/Parallel_Frames', 'data dir')
tf.app.flags.DEFINE_string(
    'architecture', 'architecture.json', 'network architecture')
tf.app.flags.DEFINE_string('logdir', 'logdir', 'log dir')
# tf.app.flags.DEFINE_string(
#     'logdir_root', None, 'log dir')
# tf.app.flags.DEFINE_string(
#     'restore_from', None, 'resotre form dir')
tf.app.flags.DEFINE_string('checkpoint', None, 'model checkpoint')

tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_float('l2_regularization', 0.0, 'L2 regularization')
tf.app.flags.DEFINE_float('lr', 1e-3, 'learning rate')
tf.app.flags.DEFINE_integer('num_steps', 10000, 'num of steps (frames)')

tf.app.flags.DEFINE_integer('source_id', 0, 'target id (SF1 = 1, TM3 = 9)')
tf.app.flags.DEFINE_integer('target_id', 9, 'target id (SF1 = 1, TM3 = 9)')

tf.app.flags.DEFINE_string(
    'file_filter', '.*\.bin', 'filename filter')

# TEST_PATTERN = 'SF1-100001.bin'
TEST_PATTERN = '.*001.bin'
N_SPEAKER = 10

mFea = 513 + 1

# ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']


def main():
    if FLAGS.checkpoint is None:
        raise

    # FLAGS
    started_datestring = "{0:%Y-%m%d-%H%M-%S}".format(datetime.now())
    logdir = os.path.join(FLAGS.logdir, 'generate', started_datestring)
    # with open(FLAGS.)

    with open(FLAGS.architecture) as f:
        architecture = json.load(f)

    # coord = tf.train.Coordinator()

    print(FLAGS.datadir)
    # label, spectrum, filename = vc2016TFWholeReader(
    #     datadir=FLAGS.datadir,
    #     pattern=TEST_PATTERN,
    #     output_filename=True)

    net = VAE2(
        batch_size=128,  # [TODO] useless attribute?
        architecture=architecture)

    x_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)

    z_ = net.encode(x_)
    xh_ = net.decode(z_, y_)

    # Restore model
    sess = tf.Session()

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(FLAGS.checkpoint))
    saver.restore(sess, FLAGS.checkpoint)
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)



    evaluator = MelCepstralProcessing(
        itype=3, order=24, isMatrix=True, drop_zeroth=True)


    # file_indices = range(150, 163)

    speakers = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']
    # src_spk = 'SF1'
    # trg_spk = 'TM3'
    src_id = FLAGS.source_id
    trg_id = FLAGS.target_id
    src_spk = speakers[src_id]
    trg_spk = speakers[trg_id]
    test_sentences = range(150, 163)
    mcds = np.zeros((len(test_sentences),))
    num_frame = np.zeros((len(test_sentences),))

    for i in test_sentences:
        src_file = os.path.join(
            FLAGS.datadir,'{}/{:6d}.mat'.format(src_spk, 100000 + i))
        trg_file = os.path.join(
            FLAGS.datadir,'{}/{}/{:6d}.mat'.format(src_spk, trg_spk, 100000 + i))
        
        # # *.bin reader
        # x = np.fromfile(src_file, dtype=np.float32)
        # x = np.reshape(x, [-1, mFea])

        x = loadmat(src_file)['sp'].T


        y = np.zeros([x.shape[0], 10])
        y[:, trg_id] = 1.0


        xh = sess.run(xh_, feed_dict={x_: x, y_: y})

        xh = np.power(10, xh)
        xh = xh / xh.sum(1).reshape([-1, 1])
        xh = xh.astype(np.float64)

        # x = np.power(10, x)
        # x = x.astype(np.float64)

        x_src = np.power(10, x)
        x_src = x_src.astype(np.float64)

        x_trg = loadmat(trg_file)['sp'].T
        x_trg = np.power(10, x_trg)
        x_trg = x_trg.astype(np.float64)

        mcd_after = evaluator.mmcd(xh, x_trg)
        mcd_before = evaluator.mmcd(x_src, x_trg)
        print('{:.2f} -> {:.2f}'.format(mcd_before, mcd_after))
        mcds[i - test_sentences[0]] = mcd_after
        num_frame[i - test_sentences[0]] = x_src.shape[0]

        # mcd.append([mcd_after, x_src.shape[0]])

        # Note:
        # 1. Note it's "log10" (so how to compute MCD?)
    mcds = num_frame * mcds 
    mcds = mcds.sum() / num_frame.sum()
    print('Mean MCD: {:.2f}'.format(mcds))

if __name__ == '__main__':
    main()
