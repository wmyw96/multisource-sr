import numpy as np
import tensorflow as tf
import time
import argparse
import sys
import os
import shutil
import importlib

from model.mssr import *
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
parser.add_argument('--outdir', default='./images/', type=str)
parser.add_argument('--exp_id', default='1', type=str)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--datadir', default='./dataset/game-live-small', type=str)
args = parser.parse_args()

# Clear out the save logs directory (for tensorboard)
if os.path.isdir(args.logdir):
    shutil.rmtree(args.logdir)

# GPU settings
if args.gpu > -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Print experiment details
print('Booting exp id: {}...'.format(args.exp_id))
time.sleep(2)

# Experiment parameters
mod = importlib.import_module('saved_params.exp'+args.exp_id)
params = mod.generate_params()

model_path = args.modeldir

# Load data

# Build the computation graph
ph, graph, graph_vars, targets, save_vars = build_mssr_model(params=params)

# Train loop
if args.gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            log_device_placement=True))
else:
    sess = tf.Session()

saver = tf.train.Saver(save_vars)
writer = tf.summary.FileWriter(args.logdir, sess.graph)

sess.run(tf.global_variables_initializer())
# restore the model here

####################
#  code to restore #
####################


saver.restore(sess, model_path)

sr_train_data, sr_valid_data, sr_test_data = sr_dataset(args.datadir, params)
small_size = (params['network']['size_h'], params['network']['size_w'])

import matplotlib.pyplot as plt

n_sources = params['network']['n_sources']

idx2name = {}
name2idx = {}
for i in range(n_sources - 1):
    data_name = params['data']['train'][i]
    name2idx[data_name] = i + 1
    idx2name[i + 1] = data_name
idx2name[0] = 'general'



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
    window_w = inp_size[1] - 2 * border
    window_h = inp_size[0] - 2 * border
    h = inp_size[0]
    w = inp_size[1]

    inp_images = []
    cover = []

    x = border
    while x < inp.shape[0]:
        y = border
        while y < inp.shape[1]:
            x1, x2 = check_inf(x - border, x - border + h, inp.shape[0])
            y1, y2 = check_inf(y - border, y - border + w, inp.shape[1])
            inp_images.append(np.expand_dims(inp[x1:x2, y1:y2, :], axis=0))
            cover.append(check_border(x1 + border, window_h, y1 + border, window_w, 
                border, inp.shape[0], inp.shape[1]))
            y += window_w
        x += window_h

    source_id = name2idx[name]

    feed_dict = {ph['lr_image']: np.concatenate(inp_images, axis=0)}
    out = sess.run(targets['samples'][source_id]['hr_fake_image'], feed_dict=feed_dict)

    sc = out.shape[1] // inp_size[0]

    out_shape = (sc * inp.shape[0], sc * inp.shape[1], inp.shape[2])
    if debug:
        print('Output shape = [{}, {}, {}]'.format(out_shape[0], out_shape[1],
            out_shape[2]))

    image = np.zeros(out_shape)
    for i in range(len(cover)):
        p = cover[i][0]
        q = cover[i][1]

        for _ in range(4):
            p[_] *= sc
            q[_] *= sc

        image[p[0]:p[1], p[2]:p[3], :] = out[i, q[0]:q[1], q[2]:q[3], :]
    return image


def print_image(sess, ph, targets, logdir, corpus_name, corpus):
    losses = []
    for _ in range(corpus.len()):
        lr_image, hr_image = corpus.get_next_batch(1)

        lr_image = lr_image[0, :, :, :]
        hr_image = hr_image[0, :, :, :]

        ts = time.time()
        out_image = get_hr_image(sess, ph, targets, name, 
            lr_image, small_size, params['data']['scale'] + 6, False)
        ts = time.time() - ts
        #print('Process image {}: {} s'.format(_, ts))

        prefix = logdir + '{}_{}'.format(corpus_name, _)

        plt.imsave(prefix + '_lr.png', lr_image)
        plt.imsave(prefix + '_hr.png', hr_image)
        plt.imsave(prefix + '_fa.png', out_image.astype(int))

        psnr_model = get_psnr(out_image, hr_image)
        psnr_cubic = 0

        #print('PSNR Cubic {}, PSNR EDSR {}'.format(psnr_cubic, psnr_model))
        losses.append(psnr_model)
    return np.mean(losses)





decay = 1.0



for name in sr_test_data:
    s = print_image(sess, ph, targets, args.outdir, name, sr_test_data[name])
    print('Dataset {}, PSNR loss = {}'.format(name, s))


