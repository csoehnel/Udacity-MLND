###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# April 12th, 2018
#
# This file contains functions logging the training process.
###############################################################################

import os
import sys
import json
import requests
import datetime
import numpy as np
from PIL import Image
from datahandling import loadImageSet
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

def initLogging(params, modelD, modelG):

    # Generate actual output folder with timestamp
    timestring = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outfolder = os.path.join(os.getcwd(), params['logdir'].split('./')[1], timestring)
    os.makedirs(outfolder, exist_ok = True)

    # Initialize TensorBoard
    tb = TensorBoard(log_dir = outfolder, batch_size = params['batchsize'], write_graph = True, write_grads = True,
                     write_images = True)
    tb.set_model(modelG)

    # Initialize checkpointing
    #mcD = ModelCheckpoint(filepath=os.path.join(outfolder, 'discriminator.epoch{epoch:04d}-epe{epe:9.5f}.h5'),
    mcD = ModelCheckpoint(filepath=os.path.join(outfolder, 'discriminator.h5'),
                          save_best_only = True, monitor='epe', mode='min')
    mcD.set_model(modelD)
    #mcG = ModelCheckpoint(filepath=os.path.join(outfolder, 'generator.epoch{epoch:04d}-epe{epe:9.5f}.h5'),
    mcG = ModelCheckpoint(filepath=os.path.join(outfolder, 'generator.h5'),
                          save_best_only = True, monitor='epe', mode='min')
    mcG.set_model(modelG)

    # Save JSON representation of models
    with open(os.path.join(outfolder, "discriminator.json"), "w") as json_fileD:
        json_fileD.write(modelD.to_json())
    with open(os.path.join(outfolder, "generator.json"), "w") as json_fileG:
        json_fileG.write(modelG.to_json())

    # Save params to file
    with open(os.path.join(outfolder, "params_train.log"), "w") as paramfile:
        paramfile.write(repr(params))

    # Clone stdout to file
    sys.stdout = Logger(sys.stdout, os.path.join(outfolder, "console_train.log"))

    return [tb, mcD, mcG, outfolder]

def logBatch(tb, nepoch, epochs, nbatch, batches, t, dt_batch, loss_D, loss_G, loss_L1):

    batchcounter = nepoch * batches + nbatch + 1
    rt = (epochs * batches - batchcounter) * dt_batch

    # Update Tensorboard logs (using on_epoch_end() for batch updates, because on_batch_end() has no effect)
    logs = {'lossD': loss_D, 'lossG': loss_G, 'lossL1': loss_L1}
    tb.on_epoch_end(nepoch, logs)

    # Time elapsed
    minutes_elapsed, seconds_elapsed = divmod(t, 60)
    hours_elapsed, minutes_elapsed = divmod(minutes_elapsed, 60)

    # Time remaining
    minutes_remaining, seconds_remaining = divmod(rt, 60)
    hours_remaining, minutes_remaining = divmod(minutes_remaining, 60)

    print("Epoch [{:4d}/{:4d}], Batch [{:4d}/{:4d}]: dt = {:8.2f}s t = {:3d}:{:02d}:{:02d} rt = {:3d}:{:02d}:{:02d} | "
          "lossD = {:9.5f} lossG = {:9.5f} lossL1 = {:9.5f}"
          .format(int(nepoch + 1), int(epochs), int(nbatch + 1), int(batches), dt_batch, int(hours_elapsed),
                  int(minutes_elapsed), int(seconds_elapsed), int(hours_remaining), int(minutes_remaining),
                  int(seconds_remaining), loss_D, loss_G, loss_L1))

def logEval(tb, mcD, mcG, nepoch, epochs, dt_eval, num_valid, epe):

    # Tensorboard update
    logs = {'epe': epe}
    tb.on_epoch_end(nepoch, logs)
    mcD.on_epoch_end(nepoch, logs)
    mcG.on_epoch_end(nepoch, logs)

    print("Epoch [{:4d}/{:4d}]: Evaluation on {:d} validation samples in {:8.2f}s ==>> EPE = {:9.5f}"
          .format(int(nepoch + 1), int(epochs), int(num_valid), dt_eval, epe))

# Saves sample image consisting of 3 vertically stacked subimages:
# 1) original image, 2) true disparity image, 3) generated disparity image
# Scaling to 8-Bit RGB values instead of original scaling for visualization.
def saveSampleImage(nepoch, nbatch, valid_paths, outImgFolder, params, fn_runG):
    [valid_img, valid_dsp] = loadImageSet(valid_paths, 0, 1, params)
    tmp = np.concatenate([fn_runG([valid_img[i:i + 1]])[0] for i in range(valid_img.shape[0])], axis = 0)[0]
    tmp = (((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))) * 255).astype(np.uint8)
    img = (((valid_img[0] - np.min(valid_img[0])) / (np.max(valid_img[0]) - np.min(valid_img[0]))) * 255).astype(np.uint8)
    dsp = (((valid_dsp[0] - np.min(valid_dsp[0])) / (np.max(valid_dsp[0]) - np.min(valid_dsp[0]))) * 255).astype(np.uint8)
    if params['color'] == 0:
        img = np.dstack([img] * 3)
        tmp = np.dstack([tmp] * 3)
        dsp = np.dstack([dsp] * 3)
    elif params['color'] == 1:
        tmp = np.dstack([tmp] * 3)
        dsp = np.dstack([dsp] * 3)
    Image.fromarray(np.vstack((np.squeeze(img), np.squeeze(dsp), np.squeeze(tmp))))\
        .save(os.path.join(outImgFolder, "epoch{:04d}-batch{:04d}.jpg".format(nepoch + 1, nbatch + 1)))

# Logging class, used to simultaneously print to stdout and to a logfile
class Logger:

    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, message):
        self.stream.write(message)
        with open(self.logfile, "a") as file:
            file.write(message)
        self.stream.flush()

    def flush(self):
        pass

# Logging class, used to send the current training status to a remote node.js tracker
class HTTPLogger:

    loggerurl = ""
    id = ""
    job = {}

    def __init__(self, machine, params):
        job = {}
        job['machine'] = machine
        job['numberOfEpochs'] = params['epochs']
        job['currentEpoch'] = 0
        job['batchesPerEpoch'] = 0
        job['currentBatch'] = 0
        job['timePerBatch'] = "0.0"
        job['timeElapsed'] = "00:00:00"
        job['timeRemaining'] = "00:00:00"
        job['lossD'] = 0.0
        job['lossG'] = 0.0
        job['lossL1'] = 0.0
        job['EPE'] = 0.0
        job['comment'] = repr(params)
        self.loggerurl = params['httploggerurl']
        url = self.loggerurl + "/add"
        res = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(self.job))
        if (res.ok):
            self.id = res.json()['_id']
            self.job = job

    def update(self, nepoch, nbatch, batches, dt_batch, t, rt, lossD, lossG, lossL1):
        if len(self.job) > 0:
            self.job['currentEpoch'] = nepoch
            self.job['batchesPerEpoch'] = batches
            self.job['currentBatch'] = nbatch
            self.job['timePerBatch'] = dt_batch
            self.job['timeElapsed'] = t
            self.job['timeRemaining'] = rt
            self.job['lossD'] = lossD
            self.job['lossG'] = lossG
            self.job['lossL1'] = lossL1
            url = self.loggerurl + "/update"
            requests.put(url, headers = {'Content-Type': 'application/json'},
                         data = json.dumps({'id': self.id, 'job': self.job}))

    def updateEPE(self, epe):
        if len(self.job) > 0:
            self.job['EPE'] = epe
            url = self.loggerurl + "/update"
            requests.put(url, headers = {'Content-Type': 'application/json'},
                         data = json.dumps({'id': self.id, 'job': self.job}))