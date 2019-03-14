import tensorflow as tf
import math


def resblock(x, n_feats, kernel_size, scale):
    tmp = tf.nn.conv2d(x, n_feats, kernel_size, activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = tf.nn.conv2d(tmp, n_feats, kernel_size, activation_fn=None)
    tmp *= scale
    return x + tmp


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
    x = tf.nn.conv2d(x, n_feats, kernel_size, activation)
    if scale & (scale - 1) == 0:    # scale = 2 ^ k
        for _ in range(int(math.log(scale, 2))):
            ps_features = 3 * (2 ** 2)
            x = tf.nn.conv2d(x, ps_features, kernel_size, activation)
            x = ps_operator(x, 2, color=True)
    else:
        ps_features = 3 * (3 ** 2)
        x = tf.nn.conv2d(x, ps_features, kernel_size, activation)
        x = ps_operator(x, 3, color=True)
    return x


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
