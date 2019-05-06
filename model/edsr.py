import tensorflow as tf
import numpy as np
from model.utils import *
import tensorflow.contrib.slim as slim


def get_edsr_ph(params):
    ph = {}

    params_d = params['network']

    lr_shape = [params_d['size_h'], params_d['size_w'], params_d['size_c']]
    scale = params_d['scale']
    hr_shape = [params_d['size_h'] * scale, params_d['size_w'] * scale,
                params_d['size_c']]

    ph['lr_image'] = tf.placeholder(dtype=tf.float32,
                                    shape=[None] + lr_shape,
                                    name='lr_image')
    ph['hr_image'] = tf.placeholder(dtype=tf.float32,
                                    shape=[None] + hr_shape,
                                    name='hr_image')
    ph['lr_decay'] = tf.placeholder(dtype=tf.float32,
                                    shape=[], name='lr_decay')
    return ph


def get_edsr_graph(ph, params):
    graph = {}

    with tf.variable_scope('edsr', reuse=False):
        inp = ph['lr_image']
        if params['network']['shift_mean']:
            inp = inp - 127

        graph['inp'] = inp

        params_d = params['data']
        params_n = params['network']

        n_feats = params_n['n_feats']
        kernel_size = params_n['kernel_size']
        graph['head'] = slim.conv2d(inp, n_feats, kernel_size)

        x = tf.identity(graph['head'])

        graph['res_block'] = []
        for i in range(params_n['n_resblocks']):
            x = resblock(x, n_feats, kernel_size, params_n['res_scale'])
            print('Block {}'.format(i) + str(x.shape))
            graph['res_block'].append(x)
        graph['body'] = \
            slim.conv2d(x, n_feats, kernel_size) + graph['head']

        up = upsampler_block(graph['body'], params_d['scale'], n_feats,
                             kernel_size, None)
        graph['hr_fake'] = up

    with tf.variable_scope('disc', reuse=False):
        print(ph['hr_image'].get_shape())
        graph['real_critic'], graph['real_feat'] = \
            build_disc(ph['hr_image'], params, reuse=False)
        graph['fake_critic'], graph['fake_feat'] = \
            build_disc(graph['hr_fake'], params, reuse=True)

        graph['hat'] = interpolate(ph['hr_image'], graph['hr_fake'])
        graph['hat_critic'], graph['hat_feat'] = \
            build_disc(graph['hat'], params, reuse=True)

    return graph


def get_edsr_targets(ph, graph, graph_vars, params):
    targets = {}

    # samples
    if params['network']['shift_mean']:
        out = graph['hr_fake'] + 127.0
    else:
        out = graph['hr_fake'] + 0.0
    samples = {'hr_fake_image': tf.clip_by_value(out, 0.0, 255.0)}

    # evaluation
    mae = tf.reduce_mean(tf.losses.absolute_difference(out,
                                                       ph['hr_image']))
    mse = tf.squared_difference(out, ph['hr_image'])
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)

    psnr = tf.constant(255**2, dtype=tf.float32) / mse
    psnr = tf.constant(10, dtype=tf.float32) * log10(psnr)

    w_dist = tf.reduce_mean(graph['real_critic']) - tf.reduce_mean(graph['fake_critic'])
    gp = tf.reduce_mean(
            (tf.sqrt(tf.reduce_sum(tf.gradients(graph['hat_critic'], graph['hat'])[0] ** 2,
                                   reduction_indices=[1, 2, 3])) - 1.) ** 2
        )
    gen_loss = -1. * tf.reduce_mean(graph['fake_critic'])
    feat_loss = tf.reduce_mean(tf.square(graph['real_feat'] - graph['fake_feat']))

    loss = mae + params['loss']['disc_loss'] * gen_loss + \
        params['loss']['feat_loss'] * feat_loss
    disc_loss = params['loss']['disc_loss'] * (-w_dist + params['loss']['gp_weight'] * gp) + \
        params['loss']['feat_loss'] * (-feat_loss)

    edsr_op = tf.train.AdamOptimizer(params['train']['lr'] * ph['lr_decay'])
    edsr_grads = edsr_op.compute_gradients(loss=loss,
                                           var_list=graph_vars['edsr'])
    edsr_train_op = edsr_op.apply_gradients(grads_and_vars=edsr_grads)

    disc_op = tf.train.AdamOptimizer(params['train']['lr'] * ph['lr_decay'])
    disc_grads = disc_op.compute_gradients(loss=disc_loss,
                                           var_list=graph_vars['disc'])
    disc_train_op = disc_op.apply_gradients(grads_and_vars=disc_grads)

    targets['train'] = {
        'mae_loss': mae,
        'mse_loss': mse,
        'psnr_loss': psnr,
        'edsr_train_op': edsr_train_op,
        'w_dist': w_dist,
        'gen_loss': gen_loss,
        'total_loss': loss
    }

    targets['train_disc'] = {
        'disc_loss': disc_loss,
        'disc_train_op': disc_train_op
    }

    targets['eval'] = {
        'mae_loss': mae,
        'mse_loss': mse,
        'psnr_loss': psnr,
    }

    targets['samples'] = samples
    return targets


def get_edsr_vars(ph, graph):
    graph_vars = {
        'edsr': tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='edsr'),
        'disc': tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='disc')
    }
    save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='edsr')
    return graph_vars, save_vars


def build_edsr_model(params):
    ph = get_edsr_ph(params)

    graph = get_edsr_graph(ph, params)
    graph_vars, save_vars = get_edsr_vars(ph, graph)
    targets = get_edsr_targets(ph, graph, graph_vars, params)

    return ph, graph, save_vars, targets
