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

log_name = "edsr"

model_path = args.modeldir + log_name + '.ckpt'
log_path = args.logdir + log_name + '.log'

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

import matplotlib.pyplot as plt
import cv2


def get_psnr(p, q):
    mse = np.mean(np.square(p - q))
    psnr = 255.0**2 / mse
    psnr = 10.0 * np.log(psnr) / np.log(10)
    return psnr



def print_image(sess, logdir, corpus_name, corpus):
    for _ in range(corpus.len() // params['train']['batch_size']):
        lr_image, hr_image = corpus.get_next_batch(
            params['train']['batch_size'])
        feed_dict = {
            ph['lr_image']: lr_image,
        }
        ts = time.time()
        out_image = sess.run(targets['samples']['hr_fake_image'], feed_dict=feed_dict)
        ts = time.time() - ts
        print('Process batch {}: {} s'.format(_, ts))
        for i in range(params['train']['batch_size']):

            prefix = logdir + '{}_{}'.format(corpus_name, 
                _ * params['train']['batch_size'] + i)

            cubic = cv2.resize(lr_image[i, :, :, :], (hr_image.shape[2], hr_image.shape[1]), 
                interpolation=cv2.INTER_CUBIC)
            plt.imsave(prefix + '_lr.png', lr_image[i, :, :, :])
            plt.imsave(prefix + '_cb.png', cubic)
            plt.imsave(prefix + '_hr.png', hr_image[i, :, :, :])
            plt.imsave(prefix + '_fa.png', out_image[i, :, :, :].astype(int))

            psnr_model = get_psnr(out_image[i, :, :, :], hr_image[i, :, :, :])
            psnr_cubic = get_psnr(cubic, hr_image[i, :, :, :])

            print('PSNR Cubic {}, PSNR EDSR {}'.format(psnr_cubic, psnr_model))




decay = 1.0



for name in sr_test_data:
    print_image(sess, args.outdir, name, sr_test_data[name])

