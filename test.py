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

####################
#  code to restore #
####################


saver.restore(sess, model_path)

sr_train_data, sr_valid_data, sr_test_data = sr_dataset(args.datadir, params)
small_size = (params['network']['size_h'], params['network']['size_w'])

import matplotlib.pyplot as plt
#import cv2


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


def get_hr_image(sess, ph, targets, inp, inp_size, border, debug=False):
    inp_images = np.expand_dims(inp, axis=0)

    feed_dict = {ph['lr_image']: inp_images}
    out = sess.run(targets['samples']['hr_fake_image'], feed_dict=feed_dict)

    return out[0, :, :, :]



def print_image(sess, ph, targets, logdir, corpus_name, corpus):
    losses = []
    for _ in range(corpus.len()):
        lr_image, hr_image = corpus.get_next_batch(1)

        lr_image = lr_image[0, :, :, :]
        hr_image = hr_image[0, :, :, :]

        ts = time.time()
        out_image = get_hr_image(sess, ph, targets, lr_image, small_size, params['data']['scale'] + 6)
        ts = time.time() - ts
        #print('Process image {}: {} s'.format(_, ts))

        prefix = logdir + '{}_{}'.format(corpus_name, _)

        #cubic = cv2.resize(lr_image, (hr_image.shape[1], hr_image.shape[0]), 
        #    interpolation=cv2.INTER_CUBIC)
        plt.imsave(prefix + '_lr.png', lr_image)
        #plt.imsave(prefix + '_cb.png', cubic)
        plt.imsave(prefix + '_hr.png', hr_image)
        plt.imsave(prefix + '_fa.png', out_image.astype(int))

        psnr_model = get_psnr(out_image, hr_image)
        #psnr_cubic = get_psnr(cubic, hr_image)
        psnr_cubic = 0.0

        #print('PSNR Cubic {}, PSNR EDSR {}'.format(psnr_cubic, psnr_model))
        losses.append(psnr_model)
    return np.mean(losses)





decay = 1.0



for name in sr_test_data:
    s = print_image(sess, ph, targets, args.outdir, name, sr_test_data[name])
    print('Dataset {}, PSNR loss = {}'.format(name, s))


