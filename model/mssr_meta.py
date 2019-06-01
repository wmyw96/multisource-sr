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

    ph['lr_image'] = {}
    ph['hr_image'] = {}

    for i in range(params['network']['n_sources']):
        ph['lr_image'][i] = tf.placeholder(dtype=tf.float32,
                                           shape=[None, None, None, 3],
                                           name='hr_image')
        ph['hr_image'][i] = tf.placeholder(dtype=tf.float32,
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
    kernel_size = params_n['kernel_size']
    branch_n_feats = params_n['n_feats_branch']

    graph['branch_feat'] = {}
    graph['shared'] = {}
    graph['inp'] = {}
    graph['global_feat'] = {}
    graph['head'] = {}
    graph['body'] = {}
    graph['hr_fake'] = {}
    graph['local_head'] = {}

    for s in range(params['network']['n_sources']):
        with tf.variable_scope('mssr-global', reuse=s > 0):
            inp = ph['lr_image'][s]
            if params['network']['shift_mean']:
                inp = inp - 127

            graph['inp'][s] = inp
            with tf.variable_scope('feat-extra', reuse=s > 0):
                graph['head'][s] = slim.conv2d(inp, n_feats, kernel_size)

            x = tf.identity(graph['head'][s])

            for _ in range(params_n['shared_block']):
                with tf.variable_scope('pre-resblock-%d' % _, reuse=s > 0):
                    x = resblock(x, n_feats, kernel_size, params_n['res_scale'])

            graph['shared'][s] = tf.identity(x)

        with tf.variable_scope('mssr-local', reuse=False):
            with tf.variable_scope('source-%d' % s, reuse=False):

                # local feat extra
                with tf.variable_scope('local-feat-extra', reuse=False):
                    graph['local_head'][s] = slim.conv2d(graph['inp'][s], branch_n_feats, kernel_size)

                lx = tf.identity(graph['local_head'][s])
                for _ in range(params_n['n_resblocks_branch_b']):
                    with tf.variable_scope('resblock-b-%d' % _, reuse=False):
                        lx = resblock(lx, branch_n_feats, kernel_size, params_n['res_scale'])                    

                lx = tf.concat([lx, graph['shared'][s]], axis=3)
                with tf.variable_scope('shared-feat-concat', reuse=False):
                    lx = slim.conv2d(lx, params_n['n_feats_branch'], kernel_size)

                for _ in range(params_n['n_resblocks_branch_a']):
                    with tf.variable_scope('resblock-a-%d' % _, reuse=False):
                        lx = resblock(lx, branch_n_feats, kernel_size, params_n['res_scale'])

                graph['branch_feat'][s] = tf.identity(lx)

        with tf.variable_scope('mssr-global', reuse=s > 0):
            x = tf.identity(graph['shared'][s])

            for _ in range(params_n['n_resblocks'] - params_n['shared_block']):
                with tf.variable_scope('aft-resblock-%d' % _, reuse=False):
                    x = resblock(x, n_feats, kernel_size, params_n['res_scale'])

            graph['global_feat'][s] = tf.identity(x)

        with tf.variable_scope('mssr-global', reuse=(s > 0)):
            #cced = tf.concat([graph['global_feat'][s], graph['branch_feat'][s]], axis=3)

            #print('Concatenated feats for source {}, shape = '.format(s) + str(cced.shape))
            #with tf.variable_scope('feat-concat', reuse=(s > 0)):
            #    graph['body'][s] = \
            #        slim.conv2d(cced, n_feats, kernel_size) + graph['head'][s]
            graph['body'][s] = graph['global_feat'][s] + graph['head'][s]
            
            with tf.variable_scope('upsamler-module', reuse=(s > 0)):
                up_gb = upsampler_block(graph['body'][s], params_d['scale'], n_feats,
                                       kernel_size, None)

        with tf.variable_scope('mssr-local', reuse=False):
            with tf.variable_scope('source-%d' % s, reuse=False):
                local_feat = graph['branch_feat'][s] + graph['local_head'][s]
                with tf.variable_scope('upsamler-module', reuse=False):
                    up_local = upsampler_block(local_feat, params_d['scale'], branch_n_feats,
                                               kernel_size, None)

                cced = tf.concat([graph['body'][s], local_feat], axis=3)
                with tf.variable_scope('local-weight', reuse=False):
                    fs = slim.conv2d(cced, branch_n_feats, kernel_size)
                    #fs = tf.nn.relu(fs)
                    local_weight = upsampler_block(fs, params_d['scale'], branch_n_feats,
                                                   kernel_size, None)
                    local_weight = tf.sigmoid(local_weight)

        up = up_gb * (1 - local_weight) + up_local * local_weight

        if params['network']['shift_mean']:
            graph['hr_fake'][s] = up + 127.0
        else:
            graph['hr_fake'][s] = up

    return graph


def get_mssr_targets(ph, graph, graph_vars, params):
    targets = {}

    targets['train_local'] = {}
    targets['samples'] = {}
    targets['eval'] = {}

    maes = []
    # samples
    for i in range(params['network']['n_sources']):
        out = tf.identity(graph['hr_fake'][i])

        samples = {'hr_fake_image': tf.clip_by_value(out, 0.0, 255.0)}

        # evaluation
        mae = tf.reduce_mean(tf.losses.absolute_difference(out,
                                                           ph['hr_image'][i]))
        mse = tf.squared_difference(out, ph['hr_image'][i])
        mae = tf.reduce_mean(mae)

        maes.append(mae)
        mse = tf.reduce_mean(mse)

        psnr = tf.constant(255**2, dtype=tf.float32) / mse
        psnr = tf.constant(10, dtype=tf.float32) * log10(psnr)

        local_op = tf.train.AdamOptimizer(params['train']['lr'] * ph['lr_decay'])
        local_grads = local_op.compute_gradients(loss=3 * mae,
                                                 var_list=graph_vars['mssr-local'][i])
        local_train_op = local_op.apply_gradients(grads_and_vars=local_grads)

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

    global_op = tf.train.AdamOptimizer(params['train']['lr'] * ph['lr_decay'])
    global_grads = global_op.compute_gradients(loss=sum(maes) / len(maes),
                                               var_list=graph_vars['mssr-global'])
    global_train_op = global_op.apply_gradients(grads_and_vars=global_grads)
    targets['train_meta'] = {
        'global_train_op': global_train_op,
    }

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

