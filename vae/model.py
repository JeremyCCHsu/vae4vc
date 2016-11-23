import tensorflow as tf

from util.layers import *

class VAE2(object):
    def __init__(
            self,
            batch_size,
            architecture):
        self.batch_size = batch_size
        self.architecture = self._make_architecture(architecture)

        # Enclose all variables in a nested dictionary
        self.variables = self._create_all_variables()

    def _make_architecture(self, nwk_arch):
        n_zy = nwk_arch['n_z'] + nwk_arch['n_y']
        enc_nwk = [nwk_arch['n_x']] + nwk_arch['encoder']['hidden'] \
            + [nwk_arch['n_z']]
        dec_nwk = [n_zy] + nwk_arch['encoder']['hidden'] \
            + [nwk_arch['n_x']]
        ver_nwk = [nwk_arch['n_x']] + nwk_arch['discriminator']['hidden'] \
            + [nwk_arch['n_d']]
        rec_nwk = [nwk_arch['n_x']] + nwk_arch['recognizer']['hidden'] \
            + [nwk_arch['n_y']]
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

                with tf.variable_scope(net):
                    if net == 'decoder':
                        n_z = nwk_arch['encoder'][-1]
                        n_y = nwk_arch['n_y']
                        n_zy = n_z + n_y

                        scope = 'z_to_zy'
                        with tf.variable_scope(scope):
                            w, b = create_weight_and_bias([n_z, n_zy])
                            var[net][scope] = {'weight': w, 'bias': b}

                        scope = 'y_to_zy'
                        with tf.variable_scope(scope):
                            w, b = create_weight_and_bias([n_y, n_zy])
                            var[net][scope] = {'weight': w, 'bias': b}
                        
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

                            layer = out + '-z'
                            with tf.variable_scope(layer):
                                w, b = create_weight_and_bias(
                                    [n_z, shapes[-1]])
                                var[net][layer] = {'weight': w, 'bias': b}
        return var

    def _encode(self, x):
        var = self.variables
        net = 'encoder'
        n_nodes = self.architecture[net]

        with tf.name_scope(net):
            for i in range(len(n_nodes) - 2):
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

        with tf.name_scope(net):
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

            for i in range(len(n_nodes) - 2):
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

    def _get_disc_feature(self, x):
        var = self.variables
        net = 'discriminator'
        n_nodes = self.architecture[net]

        with tf.name_scope(net):
            for i in range(len(n_nodes) - 2):
                layer = 'hidden{:d}'.format(i)
                with tf.name_scope(layer):
                    w = var[net][layer]['weight']
                    b = var[net][layer]['bias']
                    x = tf.nn.relu(tf.add(tf.matmul(x, w), b))
        return x

    def _verify(self, x):
        x = self._get_disc_feature(x)

        var = self.variables
        net = 'discriminator'

        with tf.name_scope(net):
            scope = 'out_mu'
            with tf.name_scope(scope):
                w = var[net][scope]['weight']
                b = var[net][scope]['bias']
                y = tf.add(tf.matmul(x, w), b)
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

            scope = 'out_mu'
            with tf.name_scope(scope):
                w = var['recognizer'][scope]['weight']
                b = var['recognizer'][scope]['bias']
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

                losses['D_KL'] = tf.reduce_mean(latent_loss)
                losses['log_p'] = tf.reduce_mean(reconstr_loss)
              

                # [Bernoulli]
                xh_mu = tf.nn.sigmoid(xh_mu)

                # logits
                pT = self._verify(x)  # + noise1)
                pF = self._verify(xh_mu)  # + noise2)

                # [TODO] This means that I SPECIFY the first dimension to be the judged TRUTH
                bz = x.get_shape().as_list()[0]
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

                # z_mu_, z_lv_ = self._encode(xh_mu)
                xh_d = self._get_disc_feature(xh_mu)
                x_d = self._get_disc_feature(x)

                # log z
                zeros = tf.zeros(tf.shape(xh_d))
                losses['info'] = - tf.reduce_mean(
                    GaussianLogDensity(x_d, xh_d, zeros, 'L_Dis'))

                # L = L_pri + L_dis + L_gan
                #   = DKL(z) + logz + gan_d

                if l2_regularization is None or l2_regularization is 0.0:
                    pass

                else:
                    l2_loss = tf.add_n([
                        tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if not('bias' in v.name)])

                    losses['l2_r'] = l2_loss * l2_regularization
                    # losses['all'] += losses['l2_r']

                for v in losses:
                    tf.scalar_summary('loss_{}'.format(v), losses[v])

                return losses
