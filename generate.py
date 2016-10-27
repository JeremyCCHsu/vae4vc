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

from datetime import datetime
from vae.model import VAE2
from iohandler.spectral_reader import vc2016TFWholeReader


FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('source', 'SF1', 'list of source speakers')
# tf.app.flags.DEFINE_string('target', 'TM3', 'list of target speakers')
# tf.app.flags.DEFINE_string('{:s}-{:s}-trn', '', 'data dir')
tf.app.flags.DEFINE_string(
    'datadir', '/home/jrm/proj/vc2016b/S-T-trn', 'data dir')
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

tf.app.flags.DEFINE_string(
    'file_filter', '.*\.bin', 'filename filter')

# TEST_PATTERN = 'SF1-100001.bin'
TEST_PATTERN = '.*001.bin'
N_SPEAKER = 10

def main():
    if FLAGS.checkpoint is None:
        raise

    # FLAGS
    started_datestring = "{0:%Y-%m%d-%H%M-%S}".format(datetime.now())
    logdir = os.path.join(FLAGS.logdir, 'generate', started_datestring)
    # with open(FLAGS.)

    with open(FLAGS.architecture) as f:
        architecture = json.load(f)

    coord = tf.train.Coordinator()

    print(FLAGS.datadir)
    label, spectrum, filename = vc2016TFWholeReader(
        datadir=FLAGS.datadir,
        pattern=TEST_PATTERN,
        output_filename=True)

    net = VAE2(
        # batch_size=spectrum.shape[0],
        batch_size=128,  # [TODO] useless?
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

    # print(1)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # print(2)
    

    # print(3)
    try:
        plt.figure(1)
        plt.figure(2)
        # z_all = list()
        idx = [30, 26, 7, 16, 13, 8, 23, 14, 5, 6, 1, 0, 10, 18, 27, 9, 12, 11, 22, 15, 31, 20, 19, 17, 3, 21, 28, 24, 4, 2, 25, 29]
        # mu = [ -2.70514116e-02,  -6.75660744e-03,  -3.11365962e+00,
        # -1.58541250e+00,  -2.01049471e+00,  -2.70431079e-02,
        # -1.15062455e-02,  -7.89349806e-03,   1.71464793e-02,
        # -1.26182893e-02,  -1.74572058e-02,   3.00247455e-03,
        #  1.97617039e-02,  -9.71486140e-03,   1.74728855e-02,
        #  1.10552078e-02,   1.47491293e-02,   5.33914950e-04,
        # -5.01557253e-03,   8.61780439e-03,  -6.06645597e-03,
        # -1.60172999e+00,  -7.61761563e-03,   1.00843329e-02,
        # -1.81709957e+00,  -2.62497878e+00,  -3.26341391e-03,
        # -9.20801703e-03,  -2.24729109e+00,  -2.02758241e+00,
        # -3.62445647e-03,  -2.74010301e-02]

        z_all = list()
        x_all = list()
        xh_all = list()
        names = list()
        id2name = [None for _ in range(10)]
        for spk in range(10):
            x_source, y_source, x_fname = sess.run([spectrum, label, filename])

            spkId = int(y_source[0])

            # tmp =  y_source[0]
            y_source = np.zeros((y_source.shape[0], 10))
            y_source[:, spkId] = 1.0

            # print(x_fname)
            x_fname = os.path.basename(x_fname)
            x_fname = os.path.splitext(x_fname)[0]
            x_fname = x_fname.decode('UTF-8')
            print(x_fname)

            tmp = re.match('(.*)-\d+', x_fname)
            id2name[spkId] = tmp.group(1)

            z = sess.run(z_, feed_dict={x_: x_source})
            xh = sess.run(xh_, feed_dict={z_: z, y_: y_source})
            # z_all.append(z)

            # pdb.set_trace()

            # z = z[:, idx]

            x_all.append(x_source)
            z_all.append(z)
            xh_all.append(xh)
            names.append(x_fname)

        tmp = np.concatenate(z_all, axis=0)
        zsd = tmp.std(0)
        idx = sorted(zip(zsd, range(len(zsd))))
        idx = [v for k, v in idx]
        # for i in range(10):
        #     plt.figure()
        #     plt.subplot(411)
        #     plt.imshow(np.flipud(x_all[i].T), aspect='auto')
        #     plt.subplots_adjust(hspace=0.001)

        #     plt.subplot(412)
        #     plt.imshow(z_all[i][:, idx].T)
        #     plt.subplots_adjust(hspace=0.001)

        #     tmp = z_all[i][:, idx]
        #     plt.subplot(414)
        #     plt.plot(tmp[:, -8:])
        #     plt.xlim([0, tmp.shape[0]])

        #     plt.subplot(413)
        #     plt.imshow(np.flipud(xh_all[i].T), aspect='auto')
        #     plt.subplots_adjust(hspace=0.001)


        #     plt.savefig('test-{:s}.png'.format(names[i]))
        #     plt.close()
            

        # [TODO] Too difficult to know the corresponding of speaker and id

        # ====== 
        # x_source = sess.run(spectrum)
        plt.figure(figsize=(48, 12))
        for spk in range(10):
            y_target = np.zeros([x_source.shape[0], N_SPEAKER])
            y_target[:, spk] = 1.0
            x_converted = sess.run(xh_, feed_dict={x_: x_source, y_: y_target})

            plt.subplot(3, 4, spk + 1)
            plt.imshow(np.flipud(x_converted.T), aspect='auto')
            # plt.axis('off')
            # plt.subplots_adjust(hspace=0.001) #, wspace=0.001)
            plt.title(id2name[spk])
        
        plt.savefig('test-{:s}-as-source.png'.format(x_fname))
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    main()
