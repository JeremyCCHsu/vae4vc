# import re
# import os
# import fnmatch
# import threading
# import librosa

import numpy as np
import tensorflow as tf

EPSILON = 1e-10


def create_weight(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


def create_weight_and_bias(shape):
    w = create_weight('weight', shape)
    b = create_bias('bias', [shape[-1]])
    return w, b


def glorot_init(shape, constant=1):
    """
    Initialization of network weights using Xavier Glorot's proposal
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    """
    _dim_sum = np.sum(shape)
    low = -constant * np.sqrt(6.0 / _dim_sum)
    high = constant * np.sqrt(6.0 / _dim_sum)
    w = tf.random_uniform(
        shape,
        minval=low,
        maxval=high,
        dtype=tf.float32)
    return w


def DenseLayer(x, fan_out, layer_name, nonlinear=tf.nn.relu):
    ''' Fully-connected (dense) layer '''
    with tf.variable_scope(layer_name):
        fan_in = x.get_shape().as_list()[-1]
        shape = (fan_in, fan_out)
        # w = tf.get_variable('w', initializer=glorot_init(shape))
        w = tf.Variable(glorot_init(shape), name='w')
        # b = tf.get_variable('b', initializer=tf.zeros([fan_out]))
        b = tf.Variable(tf.zeros([fan_out]), name='bias')
        xw = tf.matmul(x, w)
        o = tf.add(xw, b)
        y = nonlinear(o)
    return y


def MLP(var_list, fan_out, nonlinear=tf.nn.relu, is_training=False):
    ''' Note: tf.matmul only accepts 2D tensors '''
    assert isinstance(var_list, list)
    assert isinstance(fan_out, list)
    x = tf.concat(1, var_list)
    for i, n_out in enumerate(fan_out):
        x = DenseLayer(
            x, n_out,
            'DenseLayer{:0d}'.format(i + 1),
            nonlinear=nonlinear)
    return x


# def SamplingLayer(
#     mu,
#     log_var,
#     n_sample=1,
#     layer_name='SamplingLayer'):
#     ''' [WARNING] the dim order is lost; set them properly afterwards '''
#     with tf.name_scope(layer_name):
#         mu = tf.expand_dims(mu, 0)
#         std = tf.exp(log_var)
#         std = tf.sqrt(std)
#         std = tf.expand_dims(std, 0)
#         new_shape = mu.get_shape().as_list()
#         new_shape[0] = n_sample
#         eps = tf.random_normal(
#             shape=new_shape, mean=0.0, stddev=1.0, name='eps')
#         eps = tf.mul(std, eps)
#         eps = tf.add(mu, eps)
#         return eps

def mlp_to_mu_lv(x, shapes, name, reuse=True):
    with tf.variable_scope(name, reuse=reuse):
        for i in range(len(shapes) -2):
            with tf.variable_scope('layer{:d}'.format(i)):
                w = create_variable(
                    name='weight',
                    shape=shapes[i: i + 2])
                b = create_bias_variable(
                    name='bias',
                    shape=[shapes[i + 1]])
                x = tf.nn.relu(tf.add(tf.matmul(x, w), b))
        with tf.variable_scope('out_mu'):
            w = create_variable(
                name='weight',
                shape=shapes[-2:])
            b = create_bias_variable(
                name='bias',
                shape=[shapes[-1]])
            mu = tf.add(tf.matmul(x, w), b)
        with tf.variable_scope('out_lv'):
            w = create_variable(
                name='weight',
                shape=shapes[-2:])
            b = create_bias_variable(
                name='bias',
                shape=[shapes[-1]])
            lv = tf.add(tf.matmul(x, w), b)
    return mu, lv

def SamplingLayer(
        mu,
        log_var,
        # n_sample=1,
        layer_name='SamplingLayer'):
    ''' [WARNING] the dim order is lost; set them properly afterwards '''
    with tf.name_scope(layer_name):
        # mu = tf.expand_dims(mu, 0)
        std = tf.exp(log_var)
        std = tf.sqrt(std)
        # std = tf.expand_dims(std, 0)
        # new_shape = mu.get_shape().as_list()
        # new_shape[0] = n_sample
        shape = tf.shape(mu)
        # eps = tf.random_normal(
        eps = tf.truncated_normal(
            shape=shape, mean=0.0, stddev=1.0, name='eps')
        eps = tf.mul(std, eps)
        eps = tf.add(mu, eps)
        return eps


def GaussianLogDensity(x, mu, log_var, name):
    c = np.log(2 * np.pi)
    var = tf.exp(log_var)
    x_mu2 = tf.square(tf.sub(x, mu))   # [Issue] not sure the dim works or not?
    x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = tf.reduce_sum(log_prob, -1, name=name)   # keep_dims=True,
    # return tf.reduce_mean(log_prob) #, keep_dims=True
    return log_prob


def kld_of_gaussian(mu1, log_var1, mu2, log_var2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        log_var: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    var = tf.exp(log_var1)
    var2 = tf.exp(log_var2)
    mu_diff_sq = tf.square(tf.sub(mu1, mu2))
    single_variable_kld = 0.5 * (log_var2 - log_var1) \
        + 0.5 * tf.div(var, var2) * (tf.add(1.0, mu_diff_sq)) - 0.5
    return tf.reduce_sum(single_variable_kld, -1)


class VAE2(object):
    def __init__(
            self,
            batch_size,
            architecture):
        self.batch_size = batch_size
        # self.architecture = architecture

        self.architecture = self._make_architecture(architecture)

        # Enclose all variables in a nested dictionary
        self.variables = self._create_all_variables()

    # def _create_network(self, x, n_inputs, n_outputs, name):
    def _make_architecture(self, nwk_arch):
        n_zy = nwk_arch['n_z'] + nwk_arch['n_y']
        enc_nwk = [nwk_arch['n_x']] + nwk_arch['encoder']['hidden'] + [nwk_arch['n_z']]
        dec_nwk = [n_zy] + nwk_arch['encoder']['hidden'] + [nwk_arch['n_x']]
        ver_nwk = [nwk_arch['n_x']] + nwk_arch['discriminator']['hidden'] + [nwk_arch['n_d']]
        rec_nwk = [nwk_arch['n_x']] + nwk_arch['recognizer']['hidden'] + [nwk_arch['n_y']]
        return dict(
            encoder=enc_nwk,
            decoder=dec_nwk,
            discriminator=ver_nwk,
            recognizer=rec_nwk,
            n_y=nwk_arch['n_y'])


    def _create_all_variables(self):
        nwk_arch = self.architecture
        var = dict()
        with tf.variable_scope('vae'):
            for net in ['encoder', 'decoder', 'discriminator', 'recognizer']:
                var[net] = dict()
                if net == 'decoder':
                    n_z = nwk_arch['encoder'][-1]
                    n_y = nwk_arch['n_y']
                    n_zy = n_z + n_y
                    scope = 'z_to_zy'
                    with tf.variable_scope(scope):
                        # print(scope, scope.name)
                        w, b = create_weight_and_bias([n_z, n_zy])
                        var[net][scope] = {'weight': w, 'bias': b}

                    scope = 'y_to_zy'
                    with tf.variable_scope(scope):
                        w, b = create_weight_and_bias([n_y, n_zy])
                        var[net][scope] = {'weight': w, 'bias': b}
                
                with tf.variable_scope(net):
                    shapes = self.architecture[net]
                    print(net, shapes)
                    for i in range(len(shapes) -2):
                        layer = 'hidden{:d}'.format(i)
                        with tf.variable_scope(layer):
                            shape = shapes[i: i + 2]
                            w, b = create_weight_and_bias(shape)
                            var[net][layer] = {'weight': w, 'bias': b}

                        layer += '-y'
                        if net == 'decoder':
                            shape = [n_y, shapes[i + 1]]
                            w, b = create_weight_and_bias(shape)
                            var[net][layer] = {'weight': w, 'bias': b}


                    for out in ('out_mu', 'out_lv'):
                        with tf.variable_scope(out):
                            w, b = create_weight_and_bias(shapes[-2:])
                            var[net][out] = {'weight': w, 'bias': b}

                        if net == 'decoder':
                            layer = out + '-y'
                            with tf.variable_scope(layer):
                                w, b = create_weight_and_bias([n_y, shapes[-1]])
                                var[net][layer] = {'weight': w, 'bias': b}
                            # 
                            layer = out + '-z'
                            with tf.variable_scope(layer):
                                w, b = create_weight_and_bias([n_z, shapes[-1]])
                                var[net][layer] = {'weight': w, 'bias': b}
        return var

    # [TODO] _encoder and _decoder share a lot of similarity!
    # [TODO] enc and dis can be written as the same!
    def _encode(self, x):
        var = self.variables
        net = 'encoder'
        n_nodes = self.architecture[net]

        with tf.name_scope(net):
            for i in range(len(n_nodes) -2):
                layer = 'hidden{:d}'.format(i)
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    b = var[net][layer]['bias']
                    x = tf.nn.relu(tf.add(tf.matmul(x, w), b))

            outputs = list()
            for out in ['out_mu', 'out_lv']:
                with tf.name_scope(out):
                    w = var[net][out]['weight']
                    b = var[net][out]['bias']
                    y = tf.add(tf.matmul(x, w), b)
                    outputs.append(y)
        return tuple(outputs)

    def _decode(self, z, y):
        var = self.variables
        net = 'decoder'
        n_nodes = self.architecture[net]

        scope = 'z_to_zy'
        with tf.name_scope('z_to_zy'):
            w = var[net][scope]['weight']
            b = var[net][scope]['bias']
            z_to_zy = tf.matmul(z, w)

        scope = 'y_to_zy'
        with tf.name_scope('y_to_zy'):
            w = var[net][scope]['weight']
            b = var[net][scope]['bias']
            y_to_zy = tf.add(tf.matmul(y, w), b)

        x = tf.nn.relu(y_to_zy + z_to_zy)
        with tf.name_scope(net):
            for i in range(len(n_nodes) -2):
                layer = 'hidden{:d}'.format(i)
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    b = var[net][layer]['bias']
                    # y1 = tf.nn.relu(tf.add(tf.matmul(x, w), b))
                    z1 = tf.matmul(x, w)

                layer += '-y'
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    # b = var[net][layer]['bias']
                    z2 = tf.matmul(y, w)

                print(z1.get_shape(), z2.get_shape(), b.get_shape())
                x = tf.nn.relu(tf.add(z1 + z2, b))

            outputs = list()
            for out in ['out_mu', 'out_lv']:
                with tf.name_scope(out):
                    w = var[net][out]['weight']
                    b = var[net][out]['bias']
                    z_ = tf.add(tf.matmul(x, w), b)

                layer = out + '-y'
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    y_to_xh = tf.matmul(y, w)

                layer = out + '-z'
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    z_to_xh = tf.matmul(z, w)

                xh_ = tf.add(z_ + y_to_xh + z_to_xh, b)


                outputs.append(xh_)
        return tuple(outputs)

    def _verify(self, x):
        var = self.variables
        net = 'discriminator'
        n_nodes = self.architecture[net]

        with tf.name_scope(net):
            for i in range(len(n_nodes) -2):
                layer = 'hidden{:d}'.format(i)
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    b = var[net][layer]['bias']
                    x = tf.nn.relu(tf.add(tf.matmul(x, w), b))

            # outputs = list()
            # for out in 'out_mu', 'out_lv']:
            # with tf.name_scope(out):
            scope = 'out_mu'
            with tf.name_scope(scope):
                w = var[net][scope]['weight']
                b = var[net][scope]['bias']
                # y = tf.nn.softmax(tf.add(tf.matmul(x, w), b))
                y = tf.add(tf.matmul(x, w), b)
        #             outputs.append(y)
        # return tuple(outputs)
        return y


    def _recognize(self, x):
        var = self.variables
        net = 'recognizer'
        n_nodes = self.architecture[net]

        net = 'discriminator'
        with tf.name_scope('recoginzer'):
            for i in range(len(n_nodes) -2):
                layer = 'hidden{:d}'.format(i)
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    b = var[net][layer]['bias']
                    x = tf.nn.relu(tf.add(tf.matmul(x, w), b))

            # outputs = list()
            # for out in 'out_mu', 'out_lv']:
            # with tf.name_scope(out):
            scope = 'out_mu'
            with tf.name_scope(scope):
                w = var['recognizer'][scope]['weight']
                b = var['recognizer'][scope]['bias']
                # y = tf.nn.softmax(tf.add(tf.matmul(x, w), b))
                y = tf.add(tf.matmul(x, w), b)
        return y


    def encode(self, x):
        z_mu, _ = self._encode(x)
        return z_mu

    def decode(self, z, y):
        xh_mu, _ = self._decode(z, y)
        return tf.nn.sigmoid(xh_mu)

    def _compute_latent_objective(self, z_mu, z_lv):
        with tf.name_scope('latent_loss'):
            zero = tf.zeros(tf.shape(z_mu))
            Dkl = kld_of_gaussian(z_mu, z_lv, zero, zero)
            return Dkl

    def _compute_visible_objective(self, x, xh_mu, xh_lv):
        with tf.name_scope('visible_loss'):
            # logp = GaussianLogDensity(x, xh_mu, xh_lv, 'log_p_x')
            logp = tf.nn.sigmoid_cross_entropy_with_logits(xh_mu, x)
            logp = tf.reduce_sum(logp, -1)
            return logp

    def loss(self, x, y, l2_regularization=None, name='vae'):
        losses = dict()

        with tf.name_scope(name):
            z_mu, z_lv = self._encode(x)
            z = SamplingLayer(z_mu, z_lv)
            xh_mu, xh_lv = self._decode(z, y)
            xh_lv = tf.zeros(tf.shape(xh_lv))

            with tf.name_scope('loss'):
                latent_loss = self._compute_latent_objective(z_mu, z_lv)
                reconstr_loss = self._compute_visible_objective(
                    x, xh_mu, xh_lv)
                # [Watch out] L = logp - DKL, and we use a minimizer
                # loss = - tf.reduce_mean(reconstr_loss - latent_loss)

                losses['D_KL'] = tf.reduce_mean(latent_loss)
                losses['log_p'] = tf.reduce_mean(reconstr_loss)

                # loss = tf.reduce_mean(reconstr_loss) + tf.reduce_mean(latent_loss)

                losses['all'] = losses['D_KL'] + losses['log_p']
                # losses['all'] = tf.reduce_mean(latent_loss + reconstr_loss)

                

                # [Bernoulli]
                xh_mu = tf.nn.sigmoid(xh_mu)

                # logits
                pT = self._verify(x)  # + noise1)
                pF = self._verify(xh_mu)  # + noise2)

                # [TODO] This means that I SPECIFY the first dimension to be the judged TRUTH
                bz = x.get_shape().as_list()[0]
                # n_c = self.architecture['discriminator'][-1]
                l = tf.ones((bz, 1))
                o = tf.zeros((bz, 1))

                label_T = tf.concat(1, [l, o])
                label_F = tf.concat(1, [o, l])

                losses['p_t_t'] = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(pT, label_T))

                losses['p_f_f'] = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(pF, label_F))

                losses['gan_d'] = losses['p_t_t'] + losses['p_f_f']

                losses['gan_g'] = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(pF, label_T))


                # logits
                c_T = self._recognize(x)
                c_F = self._recognize(xh_mu)

                info_T_loss = tf.nn.softmax_cross_entropy_with_logits(
                    c_T,
                    y)

                info_F_loss = tf.nn.softmax_cross_entropy_with_logits(
                    c_F,
                    y)

                losses['info'] = tf.reduce_mean(info_T_loss + info_F_loss)

                # losses['p_t_t'] = tf.reduce_mean(logpTT)
                # losses['p_f_f'] = tf.reduce_mean(logpFF)
                # losses['gan_d'] = tf.reduce_mean(discriminator_loss)
                # losses['gan_g'] = tf.reduce_mean(generator_loss)
                # losses['info'] = tf.constant(0.)
                # losses['info'] = tf.reduce_mean(info_loss)

                if l2_regularization is None or l2_regularization is 0.0:
                    pass

                else:
                    l2_loss = tf.add_n([
                        tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if not('bias' in v.name)])

                    losses['l2_r'] = l2_loss * l2_regularization
                    losses['all'] += losses['l2_r']

                for v in losses:
                    tf.scalar_summary('loss_{}'.format(v), losses[v])

                return losses
