import numpy as np
import tensorflow as tf
import time
import argparse
import sys
import os
import shutil
import importlib

from model.edsr import *
from utils import *
from data import sr_ts_dataset

# os settings
sys.path.append(os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#np.set_printoptions(threshold=np.nan)
np.random.seed(0)

# Parse cmdline args
parser = argparse.ArgumentParser(description='multisource-super-resolution')
parser.add_argument('--logdir', default='./live-logs/', type=str)
parser.add_argument('--modeldir', default='./saved_models/', type=str)
parser.add_argument('--restoredir', default='./saved_models/', type=str)
parser.add_argument('--exp_id', default='1', type=str)
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

log_name = get_log_name(params, 'edsr')
model_path = args.modeldir + '/' + log_name + '.ckpt'
log_path = args.logdir + '/' + log_name + '.log'

f = open(log_path, 'w')
f.truncate()
f.close()
#with open(log_path, 'w') as f:
#    f.truncate()

# Load data
sr_ts_data = sr_ts_dataset(args.datadir, params)

# Build the computation graph
ph, graph, save_vars, targets = build_edsr_model(params=params)

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
saver.restore(sess, args.restoredir)
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



def get_hr_image(sess, ph, targets, inp, inp_size, border, debug=True):
    inp_images = np.expand_dims(inp, axis=0)
    feed_dict = {ph['lr_image']: inp_images}
    
    tt = -time.time()
    out = sess.run(targets['samples']['hr_fake_image'], feed_dict=feed_dict)
    tt += time.time()

    image = out
    return image, tt


readouts = {}
timer = TimestepSim(fps=params['live']['fps'])
timer.stop()

result = np.zeros((len(sr_ts_data), 1))

for ep in range(params['train']['num_episodes']):
    #readouts = {}

    t_ep_start = time.time()

    ts = timer.get_time()

    if (ts > len(sr_ts_data) - 1):
        break

    lr_image, hr_image = \
        sr_ts_data.get_next_batch(params['train']['batch_size'], ts, size=small_size)
    feed_dict = {
        ph['lr_image']: lr_image,
        ph['hr_image']: hr_image,
        ph['lr_decay']: decay
    }
    fetches = sess.run(targets['train'], feed_dict=feed_dict)

    timer.stop()
    print('Epoch {}, Timer = {}'.format(ep, ts))

    for k_, v_ in fetches.items():
        if 'loss' in k_:
            readouts[k_] = readouts.get(k_, []) + [fetches[k_]]

    t_ep_end = time.time()
    
    if ep % 50000 == 0 and ep > 0:
        decay *= 0.90

    ts = timer.get_time()
    timer.stop()

    eval_image = sr_ts_data.ts(ts)
    out = get_hr_image(sess, ph, targets, eval_image, None, None)
    psnr = get_psnr(out, eval_image)
    result[ts, 0] = max(psnr, result[ts, 0])

    open_file_and_save(log_path, result)
