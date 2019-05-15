import tensorflow as tf
import numpy as np
from model.utils import *
from utils import *
import tensorflow.contrib.slim as slim


def get_mssr_ph(params):
    ph = {}

    params_d = params['network']

    lr_shape = [params_d['size_h'], params_d['size_w'], params_d['size_c']]
    scale = params_d['scale']
    hr_shape = [params_d['size_h'] * scale, params_d['size_w'] * scale,
                params_d['size_c']]

    ph['lr_image'] = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None, None, 3],
                                    name='lr_image')
    ph['hr_image'] = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None, None, 3],
                                    name='hr_image')
    ph['lr_decay'] = tf.placeholder(dtype=tf.float32,
                                    shape=[], name='lr_decay')
    return ph


##########################################################################
# Implementation of Our MSSR model (MultiSource Super Resolution)
#          
#                                              resblocks
#                                               global
#    +-+     +---+      +---+           +---+          +---+
#    | |  -> |   |  ->  |   |   -> ...  |   |  -> ...  |   |
#    +-+     +---+      +---+           +---+          +---+
#    inp     feat             resblocks   |              |
#            extra             global     v            concat -> upsampler
#                                                        |
#                                       +---+          +---+
#                                       |   |  -> ...  |   |
#                                       +---+          +---+
#                                              resblocks
#                                                local
#
##########################################################################

def get_mssr_graph(ph, params):
    graph = {}
    print('Building MSSR (Multi Source Super Resolution) Comp Graph ...')

    params_d = params['data']
    params_n = params['network']

    n_feats = params_n['n_feats']
    branch_n_feats = params_n['n_feats_branch']
    kernel_size = params_n['kernel_size']

    with tf.variable_scope('mssr-global', reuse=False):
        inp = ph['lr_image']
        if params['network']['shift_mean']:
            inp = inp - 127

        graph['inp'] = inp
        with tf.variable_scope('feat-extra', reuse=False):
            graph['head'] = slim.conv2d(inp, n_feats, kernel_size)

        x = tf.identity(graph['head'])

        for i in range(params_n['start_branch']):
            with tf.variable_scope('pre-resblock-%d' % i, reuse=False):
                x = resblock(x, n_feats, kernel_size, params_n['res_scale'])

        graph['shared'] = tf.identity(x)

    graph['branch_feat'] = {}
    graph['local_head'] = {}
    with tf.variable_scope('mssr-local', reuse=False):

        for i in range(params['network']['n_sources']):
            with tf.variable_scope('source-%d' % i, reuse=False):
                with tf.variable_scope('local-feat-extra', reuse=False)
                    graph['local_head'][i] = slim.conv2d(graph['inp'], branch_n_feats, kernel_size)

                lx = tf.identity(graph['local_head'][i])
                for _ in range(params['network']['n_resblocks_branch_b']):
                    with tf.variable_scope('resblock-%d' % _, reuse=False):
                        lx = resblock(lx, branch_n_feats, kernel_size, params_n['res_scale'])                    

                lx = tf.concat([lx, graph['shared']], axis=3)
                lx = slim.conv2d(lx, params_n['n_feats_branch'], kernel_size)

                for _ in range(params_n['n_resblocks_branch_a']):
                    with tf.variable_scope('resblock-%d' % _, reuse=False):
                        lx = resblock(lx, branch_n_feats, kernel_size, params_n['res_scale'])

                graph['branch_feat'][i] = tf.identity(lx)

    with tf.variable_scope('mssr-global', reuse=False):
        x = tf.identity(graph['shared'])

        for _ in range(params_n['n_resblocks'] - params_n['start_branch']):
            with tf.variable_scope('aft-resblock-%d' % _, reuse=False):
                x = resblock(x, n_feats, kernel_size, params_n['res_scale'])

        graph['global_feat'] = tf.identity(x)

    graph['body'] = {}
    graph['hr_fake'] = {}
    for i in range(params['network']['n_sources']):

        with tf.variable_scope('mssr-global', reuse=(i > 0)):
            cced = tf.concat([graph['global_feat'], graph['branch_feat'][i]], axis=3)

            print('Concatenated feats for source {}, shape = '.format(i) + str(cced.shape))
            with tf.variable_scope('feat-concat', reuse=(i > 0)):
                graph['body'][i] = \
                    slim.conv2d(cced, n_feats, kernel_size) + graph['head']

            with tf.variable_scope('upsamler-module', reuse=(i > 0)):
                up_global = upsampler_block(graph['body'][i], params_d['scale'], n_feats,
                                            kernel_size, None)

        with tf.variable_scope('mssr-local', reuse=False):
            local_feat = graph['branch_feat'][i] + graph['local_head'][i]
            up_local = upsampler_block(local_feat, params_d['scale'], n_feats,
                                       kernel_size, None)
        up = up_global + up_local
        
        if params['network']['shift_mean']:
            graph['hr_fake'][i] = up + 127.0
        else:
            graph['hr_fake'][i] = up

    return graph


def get_mssr_targets(ph, graph, graph_vars, params):
    targets = {}

    targets['train_all'] = {}
    targets['train_local'] = {}
    targets['samples'] = {}
    targets['eval'] = {}

    # samples
    for i in range(params['network']['n_sources']):
        out = tf.identity(graph['hr_fake'][i])

        samples = {'hr_fake_image': tf.clip_by_value(out, 0.0, 255.0)}

        # evaluation
        mae = tf.reduce_mean(tf.losses.absolute_difference(out,
                                                           ph['hr_image']))
        mse = tf.squared_difference(out, ph['hr_image'])
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)

        psnr = tf.constant(255**2, dtype=tf.float32) / mse
        psnr = tf.constant(10, dtype=tf.float32) * log10(psnr)

        global_op = tf.train.AdamOptimizer(params['train']['lr'] * ph['lr_decay'])
        global_grads = global_op.compute_gradients(loss=mae,
                                                   var_list=graph_vars['mssr-global'])
        global_train_op = global_op.apply_gradients(grads_and_vars=global_grads)

        local_op = tf.train.AdamOptimizer(params['train']['lr'] * ph['lr_decay'])
        local_grads = local_op.compute_gradients(loss=mae,
                                                 var_list=graph_vars['mssr-local'][i])
        local_train_op = local_op.apply_gradients(grads_and_vars=local_grads)

        targets['train_all'][i] = {
            'mae_loss': mae,
            'mse_loss': mse,
            'psnr_loss': psnr,
            'global_train_op': global_train_op,
            'local_train_op': local_train_op
        }

        targets['train_local'][i] = {
            'mae_loss': mae,
            'mse_loss': mse,
            'psnr_loss': psnr,
            'local_train_op': local_train_op
        }

        targets['eval'][i] = {
            'mae_loss': mae,
            'mse_loss': mse,
            'psnr_loss': psnr,
        }

        targets['samples'][i] = samples

    return targets


def get_mssr_vars(ph, graph, params):
    graph_vars = {
        'mssr-global': tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='mssr-global')
    }
    graph_vars['mssr-local'] = {}
    for i in range(params['network']['n_sources']):
        graph_vars['mssr-local'][i] = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope='mssr-local/source-%d' % i)

    save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='mssr')
    return graph_vars, save_vars


def build_mssr_model(params):
    ph = get_mssr_ph(params)

    graph = get_mssr_graph(ph, params)
    graph_vars, save_vars = get_mssr_vars(ph, graph, params)
    targets = get_mssr_targets(ph, graph, graph_vars, params)

    return ph, graph, graph_vars, targets, save_vars
