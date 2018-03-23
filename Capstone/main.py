###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# December 1st, 2017
#
# This is the main file of the algorithm.
###############################################################################

import os
import time
import datetime
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils.training_utils import multi_gpu_model

from datahandling import *
from model_discriminator import *
from model_generator import *
from vislog import *

# Configure KERAS to use TENSORFLOW as backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Setup parser for command line arguments
#parser = argparse.ArgumentParser(description='Udacity Monodisparity Capstone')
#parser.add_argument('--data_path',  type=str, help='path to the data', required=True)
#parser.add_argument('--num_epochs', type=int, help='number of epochs', default=1)
#parser.add_argument('--batch_size', type=int, help='size of batch',    default=1)
#parser.add_argument('--num_gpus',   type=int, help='number of gpus',   default=1)
#args = parser.parse_args()

# Params
batchsize = 32
epochs = 12
logDir = './logs'
rt_smooth = 0.99


# Load and prepare training data
img_path_pattern = "/home/XX/FlyingThings3D/frames_cleanpass_webp/TRAIN/**/*.webp"
dsp_path_pattern = "/home/XX/FlyingThings3D/disparity/TRAIN/**/*.pfm"
print("Preparing data...", end = " ", flush = True)
t_data_start = time.time()
[train_paths, num_train] = loadFilePaths(img_path_pattern, dsp_path_pattern, True)
print("done after %4.2fs. %d training samples ready." % ((time.time() - t_data_start), num_train))

# Load and prepare validation data
img_path_pattern = "/home/XX/FlyingThings3D/frames_cleanpass_webp/TRAIN/**/*.webp"
dsp_path_pattern = "/home/XX/FlyingThings3D/disparity/TRAIN/**/*.pfm"
print("Preparing data...", end = " ", flush = True)
t_data_start = time.time()
[valid_paths, num_valid] = loadFilePaths(img_path_pattern, dsp_path_pattern, True)
print("done after %4.2fs. %d validation samples ready." % ((time.time() - t_data_start), num_valid))

### Models ###

# Set to training mode
K.set_learning_phase(1)

# Generator model
modelG = model_generator_UNet()
modelG.summary()

# Discriminator model
modelD = model_discriminator(3)
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
fn_loss = lambda output, target : K.mean(K.binary_crossentropy(output, target))

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
updatesD = Adam().get_updates(modelD.trainable_weights, [], lossD)
fn_trainD = K.function([real_img, real_dsp], [lossD / 2], updatesD)

# Function for ACTUAL TRAINING of generator
lossG = lossG_fake + lossL1
updatesG = Adam().get_updates(modelG.trainable_weights, [], lossG)
fn_trainG = K.function([real_img, real_dsp], [lossG_fake, lossL1], updatesG)

# Function for ACTUAL RUN of generator (visual/debug?)
fn_runG = K.function([real_img], [fake_dsp])

### Visualization / logging ###
tb = TensorBoard(log_dir=logDir, histogram_freq = 0, batch_size = batchsize, write_graph= False, write_grads = False,
                 write_images = False, embeddings_freq=0, embeddings_layer_names = None, embeddings_metadata = None)
tb.set_model(modelG)
outImgFolder = os.path.join(os.getcwd(), logDir.split('./')[1], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(outImgFolder, exist_ok = True)

### Training loop ###
nbatch = 0
nepoch = 0
filtered_dt = 0
batchcounter = 0

# Generator for training minibatches
train_batch = miniBatch_gf(train_paths, batchsize, True, 256, 256)
[nepoch, nbatch, train_img, train_dsp] = next(train_batch)

# Generator for validation minibatches
valid_batch = miniBatch_gf(valid_paths, 1, True, 256, 256)
[q, w, valid_img, valid_dsp] = next(valid_batch)

# Generator for training minibatches (prefetching version, multi-threaded)
#print("Prefetching training data...", end = " ", flush = True)
#bq = MiniBatchQueue(8589934592, train_paths, True, 256, 256) # Prefetching 8 GiB
#train_batch = miniBatchFromQueue_gf(bq, batchsize, len(train_paths))
#print("done. Starting training process...")

# Number of minibatches per epoch
batches = np.floor(np.float32(num_train) / np.float32(batchsize))

# Start time of training
t_training_start = time.time()

while nepoch < epochs:

    # Start time of minibatch
    t_batch_start = time.time()

    # Train discriminator
    train_lossD = fn_trainD([train_img, train_dsp])

    # Train generator
    [train_lossG, train_lossL1] = fn_trainG([train_img, train_dsp])

    # End time of minibatch
    t_batch_end = time.time()

    # Smoothing of time per batch for prediction of remaining time
    batchcounter += 1
    if batchcounter == 1:
        filtered_dt = t_batch_end - t_batch_start
    else:
        filtered_dt = (t_batch_end - t_batch_start) * (1 - rt_smooth) + filtered_dt * rt_smooth

    # Console output per batch
    print("Epoch [{:4d}/{:4d}], Batch [{:4d}/{:4d}]: dt = {:8.2f}s t = {:8.2f}s rt = {:8.2f}s | lossD = {:9.5f} lossG = {:9.5f} lossL1 = {:9.5f}"
          .format(int(nepoch + 1), int(epochs), int(nbatch + 1), int(batches), t_batch_end - t_batch_start, t_batch_end - t_training_start,
                  (epochs * batches - batchcounter) * filtered_dt, train_lossD[0], train_lossG, train_lossL1))

    # Update Tensorboard logs (using on_epoch_end() for batch updates, because on_batch_end() has no effect)
    logs = {'lossD': train_lossD[0], 'lossG': train_lossG, 'lossL1': train_lossL1}
    tb.on_epoch_end(nepoch, logs)

    # Save generator forward pass sample from validation set
    if ((batchcounter % 100) == 0):
        tmp1 = fn_runG([valid_img])[0][0]
        tmp2 = np.concatenate([fn_runG([valid_img[i:i+1]])[0] for i in range(valid_img.shape[0])], axis=0)[0]
        tmp1 = (((tmp1 - np.min(tmp1)) / (np.max(tmp1) - np.min(tmp1))) * 255).astype(np.uint8)
        tmp2 = (((tmp2 - np.min(tmp2)) / (np.max(tmp2) - np.min(tmp2))) * 255).astype(np.uint8)
        img  = (((valid_img[0] - np.min(valid_img[0])) / (np.max(valid_img[0]) - np.min(valid_img[0]))) * 255).astype(np.uint8)
        dsp  = (((valid_dsp[0] - np.min(valid_dsp[0])) / (np.max(valid_dsp[0]) - np.min(valid_dsp[0]))) * 255).astype(np.uint8)
        Image.fromarray(np.vstack((img, dsp, tmp1))).save(os.path.join(outImgFolder, "e{:04d}_b{:04d}_1.jpg".format(nepoch + 1, nbatch + 1)))
        Image.fromarray(np.vstack((img, dsp, tmp2))).save(os.path.join(outImgFolder, "e{:04d}_b{:04d}_2.jpg".format(nepoch + 1, nbatch + 1)))

    # Get next minibatch
    [nepoch, nbatch, train_img, train_dsp] = next(train_batch)
