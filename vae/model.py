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
        eps = tf.random_normal(
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
        self.architecture = architecture

        # Enclose all variables in a nested dictionary
        self.variables = self._create_all_variables()

    # def _create_network(self, x, n_inputs, n_outputs, name):

    def _create_all_variables(self):
        nwk_arch = self.architecture
        var = dict()
        with tf.variable_scope('vae'):

            # var['encoder'] = dict()
            for net in ('encoder', 'decoder'):
                var[net] = dict()
                if net == 'encoder':
                    n_nodes = [nwk_arch['n_x']] \
                        + nwk_arch[net]['hidden'] \
                        + [nwk_arch['n_z']]
                else:
                    n_nodes = [nwk_arch[net]['hidden'][0]] \
                        + nwk_arch[net]['hidden'] \
                        + [nwk_arch['n_x']]

                with tf.variable_scope(net):
                    if net == 'decoder':
                        with tf.variable_scope('y_to_zy'):
                            shape = [nwk_arch['n_y'], n_nodes[1]]
                            w = tf.Variable(glorot_init(shape), name='weight')
                            b = tf.Variable(
                                tf.zeros((shape[1])), name='bias')
                            var[net]['y_to_zy'] = {'weight': w, 'bias': b}

                        with tf.variable_scope('z_to_zy'):
                            shape = [nwk_arch['n_z'], n_nodes[1]]
                            w = tf.Variable(glorot_init(shape), name='weight')
                            b = tf.Variable(
                                tf.zeros((shape[1])), name='bias')
                            var[net]['z_to_zy'] = {'weight': w, 'bias': b}

                        with tf.variable_scope('y_to_zy_gate'):
                            shape = [nwk_arch['n_y'], n_nodes[1]]
                            w = tf.Variable(glorot_init(shape), name='weight')
                            b = tf.Variable(
                                tf.zeros((shape[1])), name='bias')
                            var[net]['y_to_zy_gate'] = {'weight': w, 'bias': b}

                        with tf.variable_scope('z_to_zy_gate'):
                            shape = [nwk_arch['n_z'], n_nodes[1]]
                            w = tf.Variable(glorot_init(shape), name='weight')
                            b = tf.Variable(
                                tf.zeros((shape[1])), name='bias')
                            var[net]['z_to_zy_gate'] = {'weight': w, 'bias': b}


                    for i, _ in enumerate(n_nodes[1: -1]):
                        layer = 'hidden{}'.format(i + 1)
                        shape = (n_nodes[i], n_nodes[i + 1])
                        # print(shape)
                        with tf.variable_scope(layer):
                            w = tf.Variable(glorot_init(shape), name='weight')
                            b = tf.Variable(
                                tf.zeros((n_nodes[i + 1])), name='bias')
                            var[net][layer] = {'weight': w, 'bias': b}

                    shape = n_nodes[-2:]
                    with tf.variable_scope('out_mu'):
                        w = tf.Variable(glorot_init(shape), name='weight')
                        b = tf.Variable(tf.zeros((n_nodes[-1])), name='bias')
                        var[net]['out_mu'] = {'weight': w, 'bias': b}

                    with tf.variable_scope('out_lv'):
                        w = tf.Variable(glorot_init(shape), name='weight')
                        b = tf.Variable(tf.zeros((n_nodes[-1])), name='bias')
                        var[net]['out_lv'] = {'weight': w, 'bias': b}

                    

        return var

    # def _f_uv(self, name, givens):
    #     with tf.variable_scope(name) as scope:
    #         n_s = self.architecture[name]['out']
    #         s0 = MLP(givens, self.architecture[name]['hidden'])
    #         s_mu = DenseLayer(s0, n_s, 'mu', tf.identity)
    #         s_log_var = DenseLayer(s0, n_s, 'log_var', tf.identity)
    #         # s_log_var = BoundLayer(
    #         #     s_log_var, LOG_VAR_FLOOR, LOG_VAR_FLOOR_lOWER)
    #     return s_mu, s_log_var

    # [TODO] _encoder and _decoder share a lot of similarity!
    def _encode(self, x):
        var = self.variables
        net = 'encoder'
        n_nodes = [self.architecture['n_x']] \
            + self.architecture[net]['hidden'] \
            + [self.architecture['n_z']]
        # print(n_nodes)
        # print(self.variables)
        # for k in self.variables:
        #     print(k, self.variables[k])
        # with tf.name_scope('vae'):
        with tf.name_scope(net):
            for i, _ in enumerate(n_nodes[1: -1]):
                layer = 'hidden{}'.format(i + 1)
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    b = var[net][layer]['bias']
                    # print(w, b, x)
                    # print(w.get_shape())
                    x = tf.nn.relu(tf.add(tf.matmul(x, w), b))
            with tf.name_scope('out_mu'):
                w = var[net]['out_mu']['weight']
                b = var[net]['out_mu']['bias']
                z_mu = tf.add(tf.matmul(x, w), b)
            with tf.name_scope('out_lv'):
                w = var[net]['out_mu']['weight']
                b = var[net]['out_mu']['bias']
                z_lv = tf.add(tf.matmul(x, w), b)
        return z_mu, z_lv

    def encode(self, x):
        z_mu, _ = self._encode(x)
        return z_mu

    def decode(self, z, y):
        xh_mu, _ = self._decode(z, y)
        return xh_mu

    def _decode(self, z, y):
        var = self.variables
        net = 'decoder'
        n_nodes = [self.architecture[net]['hidden'][0]] \
            + self.architecture[net]['hidden'] \
            + [self.architecture['n_x']]
        # with tf.variable_scope('vae'):
        with tf.name_scope(net):
            # print(y.get_shape(), z.get_shape())
            
            # x = tf.concat(1, [z, y])
            with tf.name_scope('z_to_zy'):
                w = var[net]['z_to_zy']['weight']
                b = var[net]['z_to_zy']['bias']
                z_to_zy = tf.add(tf.matmul(z, w), b)

            with tf.name_scope('y_to_zy'):
                w = var[net]['y_to_zy']['weight']
                b = var[net]['y_to_zy']['bias']
                y_to_zy = tf.add(tf.matmul(y, w), b)

            # x = tf.nn.relu(tf.add(y_to_zy, z_to_zy))
            
            x = tf.nn.tanh(tf.add(y_to_zy, z_to_zy))

            with tf.name_scope('z_to_zy_gate'):
                w = var[net]['z_to_zy_gate']['weight']
                b = var[net]['z_to_zy_gate']['bias']
                z_to_zy_gate = tf.add(tf.matmul(z, w), b)

            with tf.name_scope('y_to_zy_gate'):
                w = var[net]['y_to_zy_gate']['weight']
                b = var[net]['y_to_zy_gate']['bias']
                y_to_zy_gate = tf.add(tf.matmul(y, w), b)

            gate = tf.nn.sigmoid(tf.add(z_to_zy_gate, y_to_zy_gate))

            x = tf.mul(x, gate)

            for i, _ in enumerate(n_nodes[1: -1]):
                layer = 'hidden{}'.format(i + 1)
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    b = var[net][layer]['bias']
                    x = tf.nn.relu(tf.add(tf.matmul(x, w), b))
            with tf.name_scope('out_mu'):
                w = var[net]['out_mu']['weight']
                b = var[net]['out_mu']['bias']
                z_mu = tf.add(tf.matmul(x, w), b)
            with tf.name_scope('out_lv'):
                w = var[net]['out_mu']['weight']
                b = var[net]['out_mu']['bias']
                z_lv = tf.add(tf.matmul(x, w), b)

                # [TODO] Variance disabled.
                z_lv = tf.zeros(tf.shape(z_lv))
        return z_mu, z_lv

    def _compute_latent_objective(self, z_mu, z_lv):
        with tf.name_scope('latent_loss'):
            zero = tf.zeros(tf.shape(z_mu))
            Dkl = kld_of_gaussian(z_mu, z_lv, zero, zero)
            return Dkl

    def _compute_visible_objective(self, x, xh_mu, xh_lv):
        with tf.name_scope('visible_loss'):
            logp = GaussianLogDensity(x, xh_mu, xh_lv, 'log_p_x')
            return logp

    def loss(self, x, y, l2_regularization=None, name='vae'):
        with tf.name_scope(name):
            z_mu, z_lv = self._encode(x)
            z = SamplingLayer(z_mu, z_lv)
            xh_mu, xh_lv = self._decode(z, y)

            with tf.name_scope('loss'):
                latent_loss = self._compute_latent_objective(z_mu, z_lv)
                reconstr_loss = self._compute_visible_objective(
                    x, xh_mu, xh_lv)
                # [Watch out] L = logp - DKL, and we use a minimizer
                loss = - tf.reduce_mean(reconstr_loss - latent_loss)
                tf.scalar_summary('loss', loss)

                losses = dict()
                losses['all'] = loss
                losses['D_KL'] = tf.reduce_mean(latent_loss)
                losses['log_p'] = tf.reduce_mean(reconstr_loss)

                if l2_regularization is None or l2_regularization is 0.0:
                    # return loss
                    return losses
                else:
                    l2_loss = tf.add_n([
                        tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if not('bias' in v.name)])

                    total_loss = loss + l2_loss * l2_regularization
                    tf.scalar_summary('l2_loss', l2_loss)
                    tf.scalar_summary('total_loss', total_loss)
                    # return total_loss
                    losses['all'] = total_loss
                    return losses
