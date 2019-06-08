import numpy as np
import tensorflow as tf
import time
import argparse
import sys
import os
import shutil
import importlib
import matplotlib.pyplot as plt

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
parser.add_argument('--outdir', default='./images/', type=str)
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


def get_hr_image(sess, ph, targets, name, inp, inp_size, border, debug=True):
    inp_images = np.expand_dims(inp, axis=0)
    source_id = name2idx[name]
    feed_dict = {ph['lr_image'][source_id]: inp_images}
    
    #source_id = name2idx[name]

    tt = -time.time()
    out = sess.run(targets['samples'][source_id]['hr_fake_image'], feed_dict=feed_dict)
    tt += time.time()

    image = out[0, :, :, :]
    return image, tt


def print_image(sess, ph, targets, logdir, corpus_name, corpus):
    losses = []
    for _ in range(corpus.len()):
        lr_image, hr_image = corpus.get_next_batch(1)

        lr_image = lr_image[0, :, :, :]
        hr_image = hr_image[0, :, :, :]

        ts = time.time()
        out_image, tt = get_hr_image(sess, ph, targets, corpus_name, lr_image, small_size, params['data']['scale'] + 6, False)
        ts = time.time() - ts
        #print('Process image {}: {} s'.format(_, ts))

        prefix = logdir + '{}_{}'.format(corpus_name, _)
        #print(out_image.shape)
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

best_psnr_loss = 0.0

# restore the model here

####################
#  code to restore #
####################
saver.restore(sess, args.modeldir)



for name in sr_test_data:
    s = print_image(sess, ph, targets, args.outdir, name, sr_test_data[name])
    print('Dataset {}, PSNR loss = {}'.format(name, s))


