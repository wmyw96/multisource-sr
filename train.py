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

log_name = get_log_name(params)
model_path = args.modeldir + '/' + log_name + '.ckpt'
log_path = args.logdir + '/' + log_name + '.log'

# Load data
sr_train_data, sr_valid_data, sr_test_data = sr_dataset(args.datadir, params)

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


def evaluate(sess, sr_data):
    for name in sr_data:
        print('=' * 8 + 'Valid Set: ' + name + '=' * 8)
        corpus = sr_data[name]
        fetches = []
        for _ in range(corpus.len() // params['train']['batch_size']):
            lr_image, hr_image = corpus.get_next_batch(
                params['train']['batch_size'])
            feed_dict = {
                ph['lr_image']: lr_image,
                ph['hr_image']: hr_image,
            }
            fetch = sess.run(targets['eval'], feed_dict=feed_dict)
            fetches.append(fetch)
        print_metrics(combine_loss(fetches))


decay = 1.0

for ep in range(params['train']['num_episodes']):
    readouts = {}

    t_ep_start = time.time()

    lr_image, hr_image = \
        sr_train_data.get_next_batch(params['train']['batch_size'])
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
        readouts = {}

    if ep % 100 == 0:
        decay *= 0.9
        evaluate(sess, sr_valid_data)
        saver.save(sess, model_path)

evaluate(sess, sr_test_data)

