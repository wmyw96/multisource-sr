import tensorflow as tf
import math
import tensorflow.contrib.slim as slim


def resblock(x, n_feats, kernel_size, scale):
    tmp = slim.conv2d(x, n_feats, kernel_size, activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, n_feats, kernel_size, activation_fn=None)
    tmp *= scale
    return x + tmp


def build_disc(inp, params, reuse=False):
    with tf.variable_scope('disc-network', reuse=reuse):
        x = tf.identity(inp)
        n_feats = params['disc']['n_feats']
        kernel_size = params['disc']['kernel_size']

        for i in range(params['disc']['n_layers']):
            x = slim.conv2d(x, n_feats, kernel_size, stride=2, 
                            activation_fn=tf.nn.relu)
        sp = x.get_shape()
        feat_dim = int(sp[1] * sp[2] * sp[3])
        print('Feat Dim = {}'.format(feat_dim))
        feat = tf.reshape(x, [-1, feat_dim])
        critic = tf.layers.dense(feat, 1)
    return feat, critic


def interpolate(real, fake):
    alpha = tf.random_uniform([tf.shape(real)[0], 1, 1, 1], 0, 1, seed=123456)
    out = real * alpha + (1 - alpha) * fake
    return out


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[
        0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def ps_operator(x, r, color=False):
    if color:
        xc = tf.split(x, 3, 3)
        x = tf.concat([_phase_shift(xx, r) for xx in xc], 3)
    else:
        x = _phase_shift(x, r)
    return x


def upsampler_block(x, scale, n_feats, kernel_size, activation):
    x = slim.conv2d(x, n_feats, kernel_size, activation_fn=activation)
    if scale & (scale - 1) == 0:    # scale = 2 ^ k
        for _ in range(int(math.log(scale, 2))):
            ps_features = 3 * (2 ** 2)
            x = slim.conv2d(x, ps_features, kernel_size,
                            activation_fn=activation)
            x = ps_operator(x, 2, color=True)
    else:
        ps_features = 3 * (3 ** 2)
        x = slim.conv2d(x, ps_features, kernel_size, activation_fn=activation)
        x = ps_operator(x, 3, color=True)
    return x


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
