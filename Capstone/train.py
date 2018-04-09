###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# March 16th, 2018
#
# This is the main file for training.
# Implementation inspired by:
# https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix
# https://github.com/tjwei/GANotebooks/blob/master/pix2pix-keras.ipynb
# https://github.com/phillipi/pix2pix
###############################################################################

import os
import time
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from datahandling import *
from evaluation import *
from model_discriminator import *
from model_generator import *
from vislog import *

### Params ###
params = {
    '_comment': '',
    'batchqueuesize': 1024,
    'batchsize': 64,
    'epochs': 100,
    'httploggerurl': 'http://node:3000',
    'input_color': True,
    'input_height': 256,
    'input_normalize': 3, # 0 = none, 1 = normalization, 2 = standardization, 3 = global normalization
    'input_width': 256,
    'logdir': './logs',
    'd_beta1': 0.5,
    'd_lr': 1e-4,
    'g_beta1': 0.5,
    'g_lr': 1e-4,
    'pathpattern_img': '/home/XX/FlyingThings3D/frames_cleanpass_webp/TRAIN/**/left/*.webp',
    'pathpattern_dsp': '/home/XX/FlyingThings3D/disparity/TRAIN/**/left/*.pfm',
    'removeOutliers': True,
    'validationsplit': 0.1
}

# Configure KERAS to use TENSORFLOW as backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Set GPU to use (0 = first)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

### Setup models ###

# Set to training mode
K.set_learning_phase(1)

# Generator model
modelG = model_generator_UNet(params)

# Discriminator model
modelD = model_discriminator(3, params)

# Initialize logging
[tb, mcD, mcG, outFolder] = initLogging(params, modelD, modelG)

# Print summary of Generator
modelG.summary()

# Print summary of Discriminator
modelD.summary()

### Configuration of training process ###

# 'Real' mono image is input tensor of generator (just for shape?)
real_img = modelG.input

# 'Fake' disparity image is output tensor of generator (just for shape?)
fake_dsp = modelG.output

# 'Real' disparity image is second input tensor of discriminator (just for shape?)
real_dsp = modelD.inputs[1]

# Discriminator output for 'real' mono image and 'real' disparity image
outputD_real = modelD([real_img, real_dsp])

# Discriminator output for 'real' mono image and 'fake' disparity image
outputD_fake = modelD([real_img, fake_dsp])

# Loss function
#fn_loss = lambda output, target : K.mean(K.binary_crossentropy(output, target))
fn_loss = lambda output, target : -K.mean(K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))

# Discriminator loss, discriminator seeing 'real' mono image and 'real' disparity image
lossD_real = fn_loss(outputD_real, K.ones_like(outputD_real))

# Discriminator loss, discriminator seeing 'real' mono image and 'fake' disparity image
lossD_fake = fn_loss(outputD_fake, K.zeros_like(outputD_fake))

# Generator loss, discriminator seeing 'real' mono image and 'fake' disparity image
lossG_fake = fn_loss(outputD_fake, K.ones_like(outputD_fake))

# L1 loss
lossL1 = K.mean(K.abs(fake_dsp - real_dsp))

# Function for ACTUAL TRAINING of discriminator
lossD = lossD_real + lossD_fake
updatesD = Adam(lr = params['d_lr'], beta_1 = params['d_beta1']).get_updates(modelD.trainable_weights, [], lossD)
fn_trainD = K.function([real_img, real_dsp], [lossD / 2], updatesD)

# Function for ACTUAL TRAINING of generator
lossG = lossG_fake + 100.0 * lossL1
updatesG = Adam(lr = params['g_lr'], beta_1 = params['g_beta1']).get_updates(modelG.trainable_weights, [], lossG)
fn_trainG = K.function([real_img, real_dsp], [lossG_fake, lossL1], updatesG)

# Function for ACTUAL RUN of generator (evaluation)
fn_runG = K.function([real_img], [fake_dsp])

### Prepare and load data ###

# Load and prepare training data
print("Preparing data...", end = " ")
t_data_start = time.time()
[train_paths, num_train] = loadFilePaths(params['pathpattern_img'], params['pathpattern_dsp'], True)
print("done after {:4.2f}s. Found {:d} training samples.".format(time.time() - t_data_start, num_train))

# Determine min and max for global data normalization and outlier outlier removal
if params['input_normalize'] == 3 or params['removeOutliers']:
    print("Checking for outliers and/or global normalization boundaries...", end = " ")
    [_paths, _params] = getDatasetMinMax(train_paths, params['input_color'])
    params.update(_params)
    if params['removeOutliers']:
        train_paths = _paths
    print("done after {:4.2f}s: {:.3f}/{:.3f} (dsp) and {:.3f}/{:.3f} (img). Removed {:d} outliers.".
          format(time.time() - t_data_start, params['glob_norm_dsp_min'], params['glob_norm_dsp_max'],
                 params['glob_norm_img_min'], params['glob_norm_img_max'], num_train - len(train_paths)))
    num_train = len(train_paths)

# Validation split
print("Validation split...", end = " ")
t_data_start = time.time()
num_valid = int(np.floor(num_train * params['validationsplit']))
valid_paths = train_paths[0:num_valid]
train_paths = train_paths[num_valid:num_train]
num_train = len(train_paths)
print("done after {:4.2f}s. {:d} training samples and {:d} validation samples ready.".
      format(time.time() - t_data_start, num_train, num_valid))

# Generator for training minibatches
#train_batch = miniBatch_gf(train_paths, params)

# Generator for training minibatches (queued, multi-threaded)
print("Initializing training batch queue for {:d} samples...".format(params['batchqueuesize']), end = " ", flush = True)
tbq = MiniBatchQueue(train_paths, params)
train_batch = miniBatchFromQueue_gf(tbq, params['batchsize'], num_train)
print("done.")

# Preloading validation dataset
print("Fetching validation data...", end = " ")
t_data_start = time.time()
[valid_img, valid_dsp] = loadImageSet(valid_paths, 0, num_valid, params)
print("done after {:4.2f}s.".format(time.time() - t_data_start))

### Training loop ###

nepoch = 0

# Number of minibatches per epoch
batches = np.floor(np.float32(num_train) / np.float32(params['batchsize']))

# Start time of training / first minibatch
t_training_start = t_batch_start = time.time()

while True:

    # Start time of minibatch
    t_batch_start = time.time()

    # Get next minibatch
    [nepoch, nbatch, train_img, train_dsp] = next(train_batch)

    # Train discriminator
    train_lossD = fn_trainD([train_img, train_dsp])

    # Train generator
    [train_lossG, train_lossL1] = fn_trainG([train_img, train_dsp])

    # End time of minibatch
    t_batch_end = time.time()

    # Logging
    logBatch(tb, nepoch, params['epochs'], nbatch, batches, t_batch_end - t_training_start, t_batch_end - t_batch_start,
             train_lossD[0], train_lossG, train_lossL1)

    # End of epoch?
    if (nbatch + 1) >= batches:

        # Set to testing mode
        K.set_learning_phase(0)

        # Save sample image
        saveSampleImage(nepoch, nbatch, valid_paths, outFolder, params, fn_runG)

        # Start time of complete validation evaluation
        t_eval_start = time.time()

        # Evaluation
        epe = evaluate(valid_img, valid_dsp, params, fn_runG)

        # End time of complete validation evaluation
        t_eval_end = time.time()

        # Logging
        logEval(tb, mcD, mcG, nepoch, params['epochs'], t_eval_end - t_eval_start, num_valid, epe)

        # Set to training mode
        K.set_learning_phase(1)

    # End training?
    if nepoch >= (params['epochs'] - 1) and (nbatch >= batches - 1):
        break
