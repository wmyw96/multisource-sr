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
parser.add_argument('--exp_id', default='3', type=str)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--datadir', default='./dataset/game-live-small', type=str)
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

log_name = get_log_name(params, 'mssr')
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
for i in range(n_sources - 1):
    data_name = params['data']['train'][i]
    name2idx[data_name] = i + 1
    idx2name[i + 1] = data_name
idx2name[0] = 'general'


sess.run(tf.global_variables_initializer())
# restore the model here

####################
#  code to restore #
####################

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


def evaluate(sess, ph, targets, sr_data, mode='Valid'):
    ts = 0.0

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
            out_image = get_hr_image(sess, ph, targets, name, 
                lr_image, small_size, params['data']['scale'] + 6, False)
            ts += time.time()

            fetches.append(get_loss(out_image, hr_image))

        print('Evaluation Time: Total {} s, Mean {} s'.format(ts, ts / corpus.len()))
        ts = 0.0
        write_logs('Valid {}'.format(name), combine_loss(fetches), log_path)
        print_metrics(name, combine_loss(fetches))

        total_loss.append(summarize_loss(combine_loss(fetches)))

    return combine_loss(total_loss)


decay = 1.0

best_psnr_loss = 0.0



for ep in range(params['train']['num_episodes']):
    readouts = {}
    for s in range(n_sources):
        readouts[s] = {}

    t_ep_start = time.time()

    for source in range(n_sources):
        if source == 0:
            idd = int(np.random.randint(n_sources - 1, size=(1)))
            data_name = params['data']['train'][idd]
        else:
            data_name = params['data']['train'][source - 1]

        lr_image, hr_image = \
            sr_train_data[data_name].get_next_batch(params['train']['batch_size'], 
                size=small_size)

        feed_dict = {
            ph['lr_image']: lr_image,
            ph['hr_image']: hr_image,
            ph['lr_decay']: decay
        }
        fetches = sess.run(targets['train_all'][source], feed_dict=feed_dict)
        for k_, v_ in fetches.items():
            if 'loss' in k_:
                readouts[source][k_] = readouts[source].get(k_, []) + [fetches[k_]]


    t_ep_end = time.time()
    
    if ep % 1000 == 0:
        print('Episode: {} ({})'.format(ep, t_ep_end - t_ep_start))

        for key in readouts:
            print_metrics(idx2name[key], readouts[key])
            write_logs('Train Ep {}, {}'.format(ep, idx2name[key]), readouts[key], log_path)
        
        readouts = {}
        for s in range(n_sources):
            readouts[s] = {}
        #saver.save(sess, model_path)

    if ep % 50000 == 0 and ep > 0:
        decay *= 0.90

    if ep % 5000 == 0:
        cur_psnr_loss = np.mean(evaluate(sess, ph, targets, sr_valid_data)['psnr_loss'])

        if cur_psnr_loss > best_psnr_loss:
            best_psnr_loss = cur_psnr_loss
            saver.save(sess, model_path)

evaluate(sess, sr_test_data)

