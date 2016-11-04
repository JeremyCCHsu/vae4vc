import os
import sys
import time
import json

import tensorflow as tf
import numpy as np

from datetime import datetime
from vae.model import VAE2
from iohandler.spectral_reader import vc2016TFReader

# BATCH_SIZE = 1
# DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
# NUM_STEPS = 4000
# LEARNING_RATE = 0.02
# WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m%d-%H%M-%S}".format(datetime.now())
# SAMPLE_SIZE = 100000
# L2_REGULARIZATION_STRENGTH = 0
# SILENCE_THRESHOLD = 0.3

EPSILON = 1e-10


FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('source', 'SF1', 'list of source speakers')
# tf.app.flags.DEFINE_string('target', 'TM3', 'list of target speakers')
# tf.app.flags.DEFINE_string('{:s}-{:s}-trn', '', 'data dir')
# tf.app.flags.DEFINE_string(
#     'datadir', '/home/jrm/proj/vc2016b/S-T-trn', 'data dir')
tf.app.flags.DEFINE_string(
    'datadir', '/home/jrm/proj/vc2016b/TR_log_SP_Z_LT8000', 'data dir')

tf.app.flags.DEFINE_string(
    'architecture', 'architecture.json', 'network architecture')
tf.app.flags.DEFINE_string(
    'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'resotre form dir')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_float('l2_regularization', 0.0, 'L2 regularization')
tf.app.flags.DEFINE_float('lr', 1e-3, 'learning rate')
tf.app.flags.DEFINE_integer('num_steps', 10000, 'num of steps (frames)')
tf.app.flags.DEFINE_string(
    'file_filter', '.+(150|[01][0-4]\d|0\d\d)\.bin', 'filename filter')

# [MAKESHIFT][TODO]
tf.app.flags.DEFINE_integer('step_gan', 2000, 'num of steps before GAN')


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print('Trying to restore saved checkpoints from {} ...'.format(logdir),
        end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(
            ckpt.model_checkpoint_path
            .split('/')[-1]
            .split('-')[-1])
        print('  Global step was: {}'.format(global_step))
        print('  Restoring...', end='')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('  Done.')
        return global_step
    else:
        print(' No checkpoint found.')
        return None

#
def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }

def main():
    # validate_directories (?)

    # logdir = configs['logdir']              # jmod
    # restore_from = configs['restore_from']  # jmod
    # is_overwritten_training = lofdir != restore_from
    # restore_from = None
    try:
        directories = validate_directories(FLAGS)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    # 
    logdir = directories['logdir']
    # logdir_root = directories['logdir_root']
    restore_from = directories['restore_from']

    # # Config
    # with open(FLAGS.config, 'r') as f:
    #     configs = json.load(f)

    # Coordinator of Jobs
    coord = tf.train.Coordinator()

    # Data/Batch
    with tf.name_scope('create_input'):
        # reader = SpectralReader(
        #         ...)     # jmod [INCOMPLETE]
        # # ....
        # spectral_batch = reader.dequeue(FLAGS.batch_size)
        x, y = vc2016TFReader(
            datadir=FLAGS.datadir,
            batch_size=FLAGS.batch_size,
            num_examples_per_epoch=200000,
            num_preprocess_threads=10,
            min_fraction_of_examples_in_queue=0.005,
            pattern=FLAGS.file_filter,
            feature='log-spectrum')
        # [TODO] num_example_per_epoch: compute from datadir

    # The Learner
    with open(FLAGS.architecture) as f:
        architecture = json.load(f)

    net = VAE2(
        batch_size=FLAGS.batch_size,
        # x, y,
        architecture=architecture,
        # l2_regularization=FLAGS.l2_regularization)
        )

    # Loss and Optimizer
    # losses = net.get_losses()
    losses = net.loss(x, y, l2_regularization=FLAGS.l2_regularization)
    trainable = tf.trainable_variables()

    # [TODO] I disabled encoder update
    d_vars = [var for var in trainable if 'discriminator' in var.name]
    # print('Discriminator')
    # for v in d_vars:
    #     print(v.name)

    # print('Generator')
    g_vars = [var for var in trainable \
        if 'discriminator' not in var.name and 'recognizer' not in var.name \
        # if 'encoder' in var.name or 'decoder' in var.name \
        # and 'encoder' not in var.name
        ]


    dec_vars = [var for var in trainable if 'decoder' in var.name]

    # for v in g_vars:
    #     print(v.name)

    r_vars = [var for var in trainable if 'recognizer' in var.name]

    # Update: G(VAE+Info), D wrt GAN, R wrt Info

    for varset, name in zip(
            [g_vars, d_vars, r_vars, dec_vars], 
            ['Generator', 'Discriminator', 'Recognizer', 'Decoder']):
        print(name)
        for v in varset:
            print(v.name)
        print()



    # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)    
    # optim = optimizer.minimize(losses['all'], var_list=trainable)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)    
    optim = optimizer.minimize(
        losses['all'], 
        var_list=g_vars)

    optim_d1 = optimizer.minimize(
        losses['gan_d'],
        var_list=d_vars)

    optim_r1 = optimizer.minimize(
        -losses['info'],
        var_list=r_vars)


    # loss_infogan = 

    # optimizer_d = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, name='opt_d')
    # optim_d = optimizer_d.minimize(
    #     losses['gan_d'] - losses['info'], 
    #     var_list=list(set(d_vars + r_vars)))

    optim_d = optimizer.minimize(
        losses['gan_d'],# - losses['info'], 
        # var_list=d_vars)
        var_list=list(set(d_vars + r_vars)))

    # optimizer_d = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, name='opt_d')
    # optim_d = optimizer_d.minimize(losses['gan_d'] + losses['all'], var_list=d_vars)

    # optimizer_g = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, name='opt_g')
    # optim_g = optimizer_g.minimize(losses['gan_g'] + losses['D_KL'], var_list=g_vars)

    # optimizer_g = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, name='opt_g')
    # optim_g = optimizer_g.minimize(losses['gan_g'], var_list=g_vars)


    # optimizer_g = tf.train.AdamOptimizer(learning_rate=FLAGS.lr * 2., name='opt_g')
    
    optim_g = optimizer.minimize(
        losses['gan_g'],# - losses['info'], 
        var_list=dec_vars)

    # Writer of Summary
    writer = tf.train.SummaryWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.merge_all_summaries()

    # Session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.initialize_all_variables()
    sess.run(init)

    # Saver
    saver = tf.train.Saver()

    try:
        saved_global_step = load(saver, sess, restore_from)
        # if is_overwritten_training or saved_global_step is None:
        if saved_global_step is None:
            saved_global_step = -1
        # print(saved_global_step)
    except:
        print(
            "Something went wrong while restoring checkpoint."
            "We'll terminate training to avoid accidentally overwriting"
            " the previous models.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # reader.start_threads(sess)
    # [Q] If I command it to start_queue_runners, what's the meaning of
    # reader.start_threads?

    try:
        last_saved_step = saved_global_step
        for step in range(saved_global_step + 1, FLAGS.num_steps):
            start_time = time.time()
            if step % 50 == 0 and step < FLAGS.step_gan:
                print('Storing metadata')
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                summary, loss_value, kld, logp, _, _, _ = sess.run(
                    [summaries, losses['all'], losses['D_KL'], losses['log_p'], optim, optim_d1, optim_r1],
                    options=run_options,
                    run_metadata=run_metadata)
                writer.add_summary(summary, step)
                writer.add_run_metadata(
                    run_metadata,
                    'step_{:04d}'.format(step))
                # J: I didn't use the timeline.
            elif step >= FLAGS.step_gan:
                summary, loss_d, ptt, pff, _ = sess.run(
                    [summaries, losses['gan_d'], losses['p_t_t'], losses['p_f_f'], optim_d])
                writer.add_summary(summary, step)

                # # [TEST]
                # loss_d, ptt, pff = sess.run(
                #     [losses['gan_d'], losses['p_t_t'], losses['p_f_f']])

                summary, loss_g, _ = sess.run([summaries, losses['gan_g'], optim_g])
                summary, loss_g, _ = sess.run([summaries, losses['gan_g'], optim_g])
                writer.add_summary(summary, step)

                # # [TEST]
                # loss_g = 0
            else:
                summary, loss_value, kld, logp, _ = sess.run(
                    [summaries, losses['all'], losses['D_KL'], losses['log_p'], optim])
                writer.add_summary(summary, step)


            duration = time.time() - start_time

            if step < FLAGS.step_gan:
                print('step {:d}: D_KL = {:f}, log(p) = {:f}, ({:.3f} sec/step)'
                  .format(step, kld, logp, duration))
            else:
                # print(ptt.shape, ptt.dtype)
                print('step {:d}: D loss = {:.2f}, G\'s fooling ability = {:.2f}%, '
                    'P(T|T) = {:.2f}%, P(F|F) = {:.2f}%, ({:.3f} sec/step)'
                    .format(
                        step,
                        loss_d,
                        np.exp(-loss_g) * 100.,
                        np.exp(ptt) * 100.,
                        np.exp(pff) * 100.,
                        duration)
                )


            # if step % FLAGS.checkpoint_every == 0:
            #     save(saver, sess, logdir, step)
            #     last_saved_step = step

    except KeyboardInterrupt:
        print()

    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
