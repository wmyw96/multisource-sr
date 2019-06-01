import numpy as np
import tensorflow as tf
import time
import argparse
import sys
import os
import shutil
import importlib

from model.mssr_meta import *
from utils import *
from data import sr_dataset

# os settings
sys.path.append(os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#np.set_printoptions(threshold=np.nan)
np.random.seed(0)

# Parse cmdline args
parser = argparse.ArgumentParser(description='multisource-super-resolution')
parser.add_argument('--logdir', default='./logs', type=str)
parser.add_argument('--modeldir', default='./saved_models/', type=str)
parser.add_argument('--exp_id', default='6', type=str)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--datadir', default='./dataset/game-live', type=str)
parser.add_argument('--restore', default='', type=str)
parser.add_argument('--restoredir', default='', type=str)
args = parser.parse_args()


# Clear out the save logs directory (for tensorboard)
#if os.path.isdir(args.logdir):
#    shutil.rmtree(args.logdir)

# GPU settings
if args.gpu > -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Print experiment details
print('Booting exp id: {}...'.format(args.exp_id))
time.sleep(2)

# Experiment parameters
mod = importlib.import_module('saved_params.exp'+args.exp_id)
params = mod.generate_params()

log_name = get_log_name(params, 'mssr-meta')
model_path = args.modeldir + '/' + log_name + '.ckpt'
log_path = args.logdir + '/' + log_name + '.log'

# Clean the log file
f = open(log_path, 'w')
f.truncate()
f.close()

# Build the computation graph
ph, graph, graph_vars, targets, save_vars = build_mssr_model(params=params)

# Train loop
if args.gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            log_device_placement=True))
else:
    sess = tf.Session()

n_sources = params['network']['n_sources']
# Print Variables
show_variables('Global', graph_vars['mssr-global'])
for s_id in range(n_sources):
    show_variables('Local (Source %d)' % s_id, graph_vars['mssr-local'][s_id])

# Load data
sr_train_data, sr_valid_data, sr_test_data = sr_dataset(args.datadir, params, True)

saver = tf.train.Saver(save_vars)
writer = tf.summary.FileWriter(args.logdir, sess.graph)

idx2name = {}
name2idx = {}
for i in range(n_sources):
    data_name = params['data']['train'][i]
    name2idx[data_name] = i
    idx2name[i] = data_name
#idx2name[0] = 'general'


sess.run(tf.global_variables_initializer())

small_size = (params['network']['size_h'], params['network']['size_w'])


def get_psnr(p, q):
    mse = np.mean(np.square(p - q))
    psnr = 255.0**2 / mse
    psnr = 10.0 * np.log(psnr) / np.log(10)
    return psnr



def get_loss(p, q):
    mse = np.mean(np.square(p - q))
    mae = np.mean(np.abs(p - q))
    psnr = 255.0**2 / mse
    psnr = 10.0 * np.log(psnr) / np.log(10)
    return {
        'mse_loss': mse,
        'mae_loss': mae,
        'psnr_loss': psnr
    }


def calc_interval(left, leng, border, maxv):
    if left == border:
        le = 0
        le_s = 0
    else:
        le = left
        le_s = border

    ri = left + leng
    if ri + border < maxv:
        return le, ri, le_s, leng + border
    else:
        return le, maxv, le_s, leng + border * 2


def check_border(x1, lx, y1, ly, b, mx, my):
    xl, xr, xl_s, xr_s = calc_interval(x1, lx, b, mx)
    yl, yr, yl_s, yr_s = calc_interval(y1, ly, b, my)
    if xr_s - xl_s != xr - xl:
        print('check_border: ERROR!')
        print('[{}, {}] -> [{}, {}]'.format(xl, xr, xl_s, xr_s))
    return [xl, xr, yl, yr], [xl_s, xr_s, yl_s, yr_s]


def check_inf(l, r, mx):
    if r <= mx:
        return l, r
    else:
        return mx - (r - l), mx



def get_hr_image(sess, ph, targets, name, inp, inp_size, border, debug=True):
    inp_images = np.expand_dims(inp, axis=0)
    source_id = name2idx[name]
    feed_dict = {ph['lr_image'][source_id]: inp_images}
    
    #source_id = name2idx[name]

    tt = -time.time()
    out = sess.run(targets['samples'][source_id]['hr_fake_image'], feed_dict=feed_dict)
    tt += time.time()

    image = out
    return image, tt


def evaluate(sess, ph, targets, sr_data, mode='Valid'):
    ts = 0.0
    tt = 0.0

    total_loss = []
    for name in sr_data:
        print('=' * 8 + mode + ' set: ' + name + '=' * 8)
        corpus = sr_data[name]
        fetches = []
        for _ in range(corpus.len()):
            lr_image, hr_image = corpus.get_next_batch(1)

            lr_image = lr_image[0, :, :, :]
            hr_image = hr_image[0, :, :, :]

            ts -= time.time()
            out_image, tt_s = get_hr_image(sess, ph, targets, name, 
                lr_image, small_size, params['data']['scale'] + 6, False)
            ts += time.time()
            tt += tt_s
            fetches.append(get_loss(out_image, hr_image))

        print('Evaluation Time: Total {} s, Mean {} s'.format(ts, ts / corpus.len()))
        print('Tensorflow Time: Total {} s, Mean {} s'.format(tt, tt / corpus.len()))

        ts = 0.0
        tt = 0.0
        write_logs('Valid {}'.format(name), combine_loss(fetches), log_path)
        print_metrics(combine_loss(fetches), name)

        total_loss.append(summarize_loss(combine_loss(fetches)))

    return combine_loss(total_loss)


decay = 1.0

best_psnr_loss = 0.0

# restore the model here

####################
#  code to restore #
####################

begin_ep = 0

import re

if len(args.restore) > 0:
    log_file_path_old = args.logdir + '/' + args.restore + '.log'
    with f.open(log_file_path_old, 'r') as f:
        ep = 0
        pattern = re.compile(r'(?<=Train Ep =)\d+\.?\d*')
        for line in f.readlines():
            if 'Train Ep' in line:
                ep = int(pattern.findall(line)[0])
            elif 'Valid' in line:
                begin_ep = max(begin_ep, ep)
            else:
                assert Exception, "Invalid log file"
    model_path = args.restoredir
    saver.restore(sess, restoredir)
    for i in range(begin_ep // params['train']['decay_interval']):
        decay *= 0.9


readouts = {}
for s in range(n_sources):
    readouts[s] = {}


for ep in range(begin_ep, params['train']['num_episodes']):

    t_ep_start = time.time()

    for s in range(n_sources):
        data_name = params['data']['train'][s]

        lr_image, hr_image = \
            sr_train_data[data_name].get_next_batch(params['train']['batch_size'], 
                size=small_size)

        feed_dict = {
            ph['lr_image'][s]: lr_image,
            ph['hr_image'][s]: hr_image,
            ph['lr_decay']: decay
        }
        fetches = sess.run(targets['train_local'][s], feed_dict=feed_dict)
        for k_, v_ in fetches.items():
            if 'loss' in k_:
                readouts[s][k_] = readouts[s].get(k_, []) + [fetches[k_]]

    feed_dict = {ph['lr_decay']: decay}
    for s in range(n_sources):
        data_name = params['data']['train'][s]

        lr_s, hr_s = \
            sr_train_data[data_name].get_next_batch(params['train']['meta_batch_size'], 
                size=small_size)
        feed_dict[ph['lr_image'][s]] = lr_s
        feed_dict[ph['hr_image'][s]] = hr_s
    fetches = sess.run(targets['train_meta'], feed_dict=feed_dict)

    t_ep_end = time.time()
    
    if ep % params['train']['trainlog_interval'] == 0:
        print('Episode: {} ({})'.format(ep, t_ep_end - t_ep_start))

        for key in readouts:
            print_metrics(readouts[key], idx2name[key])
            write_logs('Train Ep {}, {}'.format(ep, idx2name[key]), readouts[key], log_path)
        
        readouts = {}
        for s in range(n_sources):
            readouts[s] = {}
        #saver.save(sess, model_path)

    if ep % params['train']['decay_interval'] == 0 and ep > 0:
        decay *= 0.90

    if ep % params['train']['validlog_interval'] == 0:
        cur_psnr_loss = np.mean(evaluate(sess, ph, targets, sr_valid_data)['psnr_loss'])

        if cur_psnr_loss > best_psnr_loss:
            best_psnr_loss = cur_psnr_loss
            saver.save(sess, model_path)

evaluate(sess, sr_test_data)


