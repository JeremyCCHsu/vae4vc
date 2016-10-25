import os
import sys
import time
import json

import tensorflow as tf

from datetime import datetime
from vae.model import VAE2
from iohandler.spectral_reader import vc2016TFReader


def main():
	# FLAGS
	started_datestring = "{0:%Y-%m%d-%H%M-%S}".format(datetime.now())
	logdir = os.path.join(FLAGS.logdir, 'generate', started_datestring)
	# with open(FLAGS.)

	with open(FLAGS.architecture) as f:
	    architecture = json.load(f)

	sess = tf.Session()

	net = VAE2(batch_size, architecture=architecture)

	variables_to_restore = {
	    var.name[:-2]: var for var in tf.all_variables()
	    if not ('state_buffer' in var.name or 'pointer' in var.name)}
	saver = tf.train.Saver(variables_to_restore)

	print('Restoring model from {}'.format(args.checkpoint))
	saver.restore(sess, args.checkpoint)

