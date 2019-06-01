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

# Build the computation graph
ph, graph, save_vars, targets = build_edsr_model(params=params)

# Load data
sr_train_data, sr_valid_data, sr_test_data = sr_dataset(args.datadir, params)

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
            out_image, stt = get_hr_image(sess, ph, targets, lr_image, small_size, params['data']['scale'] + 6, False)
            ts += time.time()
            tt += stt

            fetches.append(get_loss(out_image, hr_image))

        print('Tensorflow Time: Total {} s, Mean {} s'.format(tt, tt / corpus.len()))
        print('Evaluation Time: Total {} s, Mean {} s'.format(ts, ts / corpus.len()))
        ts = 0.0
        write_logs('Valid {}'.format(name), combine_loss(fetches), log_path)
        print_metrics(combine_loss(fetches))

        total_loss.append(summarize_loss(combine_loss(fetches)))

    return combine_loss(total_loss)


decay = 1.0

best_psnr_loss = 0.0

readouts = {}

for ep in range(params['train']['num_episodes']):
    #readouts = {}

    t_ep_start = time.time()
    steps = params['train']['disc_steps']
    if ep < 100 or ep % 100 == 0:
        steps = 100
    # train disc
    for k in range(steps):
        lr_image, hr_image = \
            sr_train_data.get_next_batch(params['train']['batch_size'], size=small_size)
        feed_dict = {
            ph['lr_image']: lr_image,
            ph['hr_image']: hr_image,
            ph['lr_decay']: decay
        }
        _ = sess.run(targets['train_disc'], feed_dict=feed_dict)
        print('Train Disc: Wdist = {}'.format(_['wdist_loss']))
    
    lr_image, hr_image = \
        sr_train_data.get_next_batch(params['train']['batch_size'], size=small_size)
    feed_dict = {
        ph['lr_image']: lr_image,
        ph['hr_image']: hr_image,
        ph['lr_decay']: decay
    }
    fetches = sess.run(targets['train'], feed_dict=feed_dict)
    for k_, v_ in fetches.items():
        if 'loss' in k_:
            readouts[k_] = readouts.get(k_, []) + [fetches[k_]]

    t_ep_end = time.time()
    
    if ep % 100 == 0:
        print('Episode: {} ({})'.format(ep, t_ep_end - t_ep_start))
        print_metrics(readouts)
        write_logs('Train Ep {}'.format(ep), readouts, log_path)
        readouts = {}
        #saver.save(sess, model_path)

    if ep % 50000 == 0 and ep > 0:
        decay *= 0.90

    if ep % 5000 == 0:
        #decay *= 0.90
        #print(evaluate(sess, ph, targets, sr_valid_data)['psnr_loss'])
        cur_psnr_loss = np.mean(evaluate(sess, ph, targets, sr_valid_data)['psnr_loss'])

        if cur_psnr_loss > best_psnr_loss:
            best_psnr_loss = cur_psnr_loss
            saver.save(sess, model_path)

evaluate(sess, ph, targets, sr_test_data)

