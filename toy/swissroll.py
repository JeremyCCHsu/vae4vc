import pdb
import tensorflow as tf
import numpy as np
# import matplotlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


EPSILON = 1e-10

# N = 10000
# u1, u2 = 10., 0.
# s1, s2 = 3., 0.2
# v1 = np.random.normal(u1, s1, (N, 1))
# v2 = np.random.normal(u2, s2, (N, 1))

# r = 0.4 * v1 + v2
# y = np.concatenate(
#   [r * np.cos(v1), r * np.sin(v1)],
#   axis=1)


# plt.figure()
# # plt.plot(y.T, 'x')
# plt.plot(y[:,0], y[:,1], 'x')
# plt.savefig('test.png')
# plt.close()


def swissroll(N, u1, s1, u2, s2):
    v1 = np.random.normal(u1, s1, (N, 1))
    v2 = np.random.normal(u2, s2, (N, 1))

    r = 0.4 * v1 + v2
    y = np.concatenate(
        [r * np.cos(v1), r * np.sin(v1)],
        axis=1)

# G & D: 2-layered 64 units

# def main():

def create_weight(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer()
    variable = tf.get_variable(
        initializer=initializer(shape=shape),
        name=name)
    return variable


def create_bias(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.get_variable(
        initializer=initializer(shape=shape),
        name=name)


def create_weight_and_bias(shape):
    w = create_weight('weight', shape)
    b = create_bias('bias', [shape[-1]])
    return w, b


# 
def swissroll(N, u1, s1, u2, s2):
    v1 = tf.random_normal((N, 1), u1, s1)
    v2 = tf.random_normal((N, 1), u2, s2)

    r = 0.4 * v1 + v2
    y = tf.concat(
        1,
        [r * tf.cos(v1), r * tf.sin(v1)])
    return y

def swissroll_bounded(N, u1, s1, u2, s2):
    y = swissroll(N, u1, s1, u2, s2)
    y = tf.maximum(y, -8.)
    y = tf.minimum(y, 8.)
    y = tf.div(y, 8.)
    return y

class ToyGAN(object):
    # u1, u2 = 10., 0.
    # s1, s2 = 3., 0.2
    def __init__(self, N=128, u1=10., s1=3., u2=0., s2=.2):
        with tf.variable_scope('x'):
            x = swissroll_bounded(N, u1, s1, u2, s2)
            # x = swissroll(N, u1, s1, u2, s2)

        with tf.variable_scope('z'):
            z = tf.random_uniform((128, 2), -1., 1., name='z')
            # print(x.get_shape())
            # z = tf.reduce_mean(x, 1, keep_dims=True)

        with tf.variable_scope('generator'):
            xh = self._generate(z)

        with tf.name_scope('noise'):
            n = tf.random_normal((128, 2), 0., 0.5)

        with tf.variable_scope('discriminator'):
            p, ph = self._discriminate(x, xh)

        with tf.name_scope('loss'):
            self.losses = self._loss(p, ph)

            # self.losses.update({'mse': self._loss_mse(x, xh)})
            self.losses.update(self._loss_mse(x, xh))

        self.x = x
        self.xh = xh
        self.p = p

    def _generate(self, x):
        with tf.variable_scope('layer1'):
            w, b = create_weight_and_bias((2, 64))
            x = tf.nn.relu(tf.add(tf.matmul(x, w), b))

        with tf.variable_scope('layer2'):
            w, b = create_weight_and_bias((64, 64))
            x = tf.nn.relu(tf.add(tf.matmul(x, w), b))

        with tf.variable_scope('out'):
            w, b = create_weight_and_bias((64, 2))
            x = tf.nn.tanh(tf.add(tf.matmul(x, w), b))
            # x = tf.add(tf.matmul(x, w), b)

        return x


    def _discriminate(self, x, xh):
        with tf.variable_scope('layer1'):
            w, b = create_weight_and_bias((2, 64))
            x = tf.nn.relu(tf.add(tf.matmul(x, w), b))

        with tf.variable_scope('layer2'):
            w, b = create_weight_and_bias((64, 64))
            x = tf.nn.relu(tf.add(tf.matmul(x, w), b))

        with tf.variable_scope('out'):
            w, b = create_weight_and_bias((64, 1))
            x = tf.nn.sigmoid(tf.add(tf.matmul(x, w), b))

        # ==== For fake ====
        with tf.variable_scope('layer1', reuse=True):
            w, b = create_weight_and_bias((2, 64))
            xh = tf.nn.relu(tf.add(tf.matmul(xh, w), b))

        with tf.variable_scope('layer2', reuse=True):
            w, b = create_weight_and_bias((64, 64))
            xh = tf.nn.relu(tf.add(tf.matmul(xh, w), b))

        with tf.variable_scope('out', reuse=True):
            w, b = create_weight_and_bias((64, 1))
            xh = tf.nn.sigmoid(tf.add(tf.matmul(xh, w), b))

        return x, xh

    def _loss(self, p, ph):
        # pTT = p[:, 0]
        # pFF = ph[:, 1]
        # pTF = ph[:, 0]
        pTT = p
        # pFF = 1. - ph
        pTF = ph
        pFF = 1. - ph
        # with tf.name_scope('loss'):
        loss_d = - (tf.log(pTT + EPSILON) + tf.log(pFF + EPSILON))
        # loss_g = -tf.log(pTF + EPSILON)
        # loss_g = - tf.log(tf.div(pTF, pFF + EPSILON) + EPSILON)
        loss_g = - (tf.log(pTF + EPSILON) - tf.log(pFF + EPSILON))
        # loss_g = - tf.log(tf.div(pTF, pFF + EPSILON) + EPSILON)

        return dict(
            loss_d=tf.reduce_mean(loss_d),
            loss_g=tf.reduce_mean(loss_g))

    # def _loss_mse(self, x, xh):
    #     ''' this MSE is WRONG because z and x aren't aligned!
    #         That's why Eric Jang proposed a sorted version in his blog.
    #         http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
    #     '''
    #     return {'mse': tf.reduce_mean(tf.reduce_sum(tf.abs(x - xh), 1))}

        # return 
    def get_losses(self):
        return self.losses

    def get_x(self):
        return self.x

    def get_xh(self):
        return self.xh

    # def predict(self, x):
    #   return sess.run(p, feed_dict={self.x: x})



def main():  # trian
    gan = ToyGAN()
    # N = 50,000 / 128 = 390

    losses = gan.get_losses()
    trainable = tf.trainable_variables()

    d_vars = [v for v in trainable if 'discriminator' in v.name]
    g_vars = [v for v in trainable if 'generator' in v.name]

    lr = 2e-4
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    opt_d = opt.minimize(losses['loss_d'], var_list=d_vars)
    
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    opt_g = opt.minimize(losses['loss_g'], var_list=g_vars)

    opt_m = opt.minimize(losses['mse'], var_list=g_vars)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    writer = tf.train.SummaryWriter('test_log')
    writer.add_graph(tf.get_default_graph())
    for epoch in range(200):
        for i in range(400):
            Ld, _ = sess.run([losses['loss_d'], opt_d], feed_dict={})
            Lg, _ = sess.run([losses['loss_g'], opt_g], feed_dict={})
            # Lg, _ = sess.run([losses['loss_g'], opt_g], feed_dict={})
            print('Iter {:2}-{:3d}: Ld: {}, Lg: {}'.format(epoch, i, Ld, Lg))

            # Lm, _ = sess.run([losses['mse'], opt_m])
            # print('Iter {}-{}: Lm = {}'.format(epoch, i, Lm))

    xh = list()
    x = list()
    p = list()
    for _ in range(400):
        # xh_ = sess.run(gan.get_xh())
        xh.append(sess.run(gan.get_xh()))

        x_ = sess.run(gan.get_x())
        x.append(x_)

        p_ = sess.run(gan.p, feed_dict={gan.x: x_})
        p.append(p_)

    xh = np.concatenate(xh, 0)
    x = np.concatenate(x, 0)

    plt.figure()
    plt.plot(x[:, 0], x[:, 1], 'ro')
    plt.hold(True)
    plt.plot(xh[:, 0], xh[:, 1], 'x')
    plt.savefig('test-x-xh.png')
    plt.close()
    
    # p = np.concatenate(p, 0)
    # plt.figure()
    # plt.plot(xh[p[:,0]>.5, 0], xh[p[:,0]>.5, 1], 'ro')
    # plt.hold(True)
    # plt.plot(xh[p[:,0]<=.5, 0], xh[p[:,0]<=.5, 1], 'bx')
    # plt.savefig('test-p.png')
    # plt.close()
    

    # z = [[x, y] for x in np.linspace(-1, 1, 256) for y in np.linspace(-1, 1, 256)]
    # z = np.asarray(z)

    saver = tf.train.Saver()
    saver.save(sess, 'test_log/model.ckpt', global_step=epoch)

    x = list()
    p = list()
    for i in np.linspace(-1, 1, 128):
        x_ = [i * np.ones((128,)), np.linspace(-1, 1, 128)]
        x_ = np.asarray(x_).T
        p_ = sess.run(gan.p, feed_dict={gan.x: x_})
        x.append(x_)
        p.append(p_)

    p = np.concatenate(p, axis=0)
    x = np.concatenate(x, axis=0)

    pdb.set_trace()

    # isTrue = p[:]
    plt.figure()
    plt.plot(x[p[:, 0] > .5, 0], x[p[:, 0] > .5, 1], 'ro')
    plt.hold(True)
    plt.plot(x[p[:, 0] <= .5, 0], x[p[:, 0] <= .5, 1], 'bx')
    plt.savefig('test-all-area.png')

    print('Generator')
    for v in g_vars:
        print(v.name)
    
    # 
    print('\nDiscriminator')
    for v in d_vars:
        print(v.name)

if __name__ == '__main__':
    main()
