import numpy as np
import tensorflow as tf

EPSILON = 1e-6

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
    return tf.Variable(initializer(shape=shape), name=name)


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
        w = tf.Variable(glorot_init(shape), name='w')
        b = tf.Variable(tf.zeros([fan_out]), name='bias')
        xw = tf.matmul(x, w)
        o = tf.add(xw, b)
        y = nonlinear(o)
    return y


def SamplingLayer(
        mu,
        log_var,
        # n_sample=1,
        layer_name='SamplingLayer'):
    ''' [WARNING] the dim order is lost; set them properly afterwards '''
    with tf.name_scope(layer_name):
        std = tf.exp(log_var)
        std = tf.sqrt(std)
        shape = tf.shape(mu)
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
