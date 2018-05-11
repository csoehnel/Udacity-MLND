###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# May 11th, 2018
#
# This file contains functions for serving data.
###############################################################################

import sys
import cv2
import glob
import time
import copy
import random
import threading
import numpy as np
from collections import deque
from PIL import Image
from python_pfm import *

def loadSample(img_path, dsp_path, params):

    sample_img = cv2.imread(img_path)
    [sample_dsp, scale] = readPFM(dsp_path)

    # Color or Grey-Scale
    if params['color'] == 0:
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2GRAY)
    elif params['color'] == 1:
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2RGB)
    elif params['color'] == 2:
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2RGB)
        sample_dsp = np.dstack([sample_dsp] * 3)

    # Resizing
    if params['input_width'] >= 1 and params['input_height'] >= 1:
        # INTER_AREA best for shrinkage
        sample_img = cv2.resize(sample_img, (params['input_width'], params['input_height']),
                                interpolation = cv2.INTER_AREA).astype('float32')
        sample_dsp = cv2.resize(sample_dsp, (params['input_width'], params['input_height']),
                                interpolation = cv2.INTER_AREA).astype('float32')
    else:
        sample_img = sample_img.astype('float32')
        sample_dsp = sample_dsp.astype('float32')

    # Reshape axes
    if params['color'] == 0:
        sample_img = sample_img[:, :, np.newaxis]
        sample_dsp = sample_dsp[:, :, np.newaxis]
    elif params['color'] == 1:
        sample_dsp = sample_dsp[:, :, np.newaxis]

    # Scaling
    if params['input_normalize'] == 1: # Normalize
        sample_img = (sample_img - np.min(sample_img)) / (np.max(sample_img) - np.min(sample_img))
        sample_dsp = (sample_dsp - np.min(sample_dsp)) / (np.max(sample_dsp) - np.min(sample_dsp))
    elif params['input_normalize'] == 2: # Standardize
        sample_img = (sample_img - np.mean(sample_img)) / np.std(sample_img)
        sample_dsp = (sample_dsp - np.mean(sample_dsp)) / np.std(sample_dsp)
    elif params['input_normalize'] == 3: # Global normalization
        sample_img = (sample_img - params['glob_norm_img_min']) / (params['glob_norm_img_max'] -
                                                                   params['glob_norm_img_min'])
        sample_dsp = (sample_dsp - params['glob_norm_dsp_min']) / (params['glob_norm_dsp_max'] -
                                                                   params['glob_norm_dsp_min'])

    return [sample_img, sample_dsp]

def loadFilePaths(img_path_pattern, dsp_path_pattern, randomize = False):

    # Find file paths
    img_paths = glob.glob(img_path_pattern, recursive=True)
    dsp_paths = glob.glob(dsp_path_pattern, recursive=True)

    # Prepare lists for checks
    img_path_pattern_prefix = img_path_pattern.split('*')[0]
    dsp_path_pattern_prefix = dsp_path_pattern.split('*')[0]
    img_path_pattern_suffix = img_path_pattern.split('*')[-1]
    dsp_path_pattern_suffix = dsp_path_pattern.split('*')[-1]
    img_paths = [img_path.split(img_path_pattern_prefix)[1].split(img_path_pattern_suffix)[0] for img_path in img_paths]
    dsp_paths = [dsp_path.split(dsp_path_pattern_prefix)[1].split(dsp_path_pattern_suffix)[0] for dsp_path in dsp_paths]

    # Check lists on missing entries on either side
    vld_paths = list(set(img_paths) & set(dsp_paths))

    # Rebuild valid lists
    img_paths = [img_path_pattern_prefix + vld_path + img_path_pattern_suffix for vld_path in vld_paths]
    dsp_paths = [dsp_path_pattern_prefix + vld_path + dsp_path_pattern_suffix for vld_path in vld_paths]
    img_paths.sort()
    dsp_paths.sort()

    # Randomize entries
    cmb_paths = list(zip(img_paths, dsp_paths))
    if randomize:
        random.shuffle(cmb_paths)

    return [cmb_paths, len(cmb_paths)] # return list with consistent file paths

def loadImageSet(paths, i, num2load, params):

    img = []
    dsp = []
    num2load = min(i + num2load, len(paths))
    for j in range(i, i + num2load):
        [sample_img, sample_dsp] = loadSample(paths[j][0], paths[j][1], params)
        img.append(sample_img)
        dsp.append(sample_dsp)
    return [np.float32(img), np.float32(dsp)]

def getDatasetMinMax(paths, color):

    _params = {
        'glob_norm_dsp_min': np.inf,
        'glob_norm_dsp_max': 0,
        'glob_norm_img_min': np.inf,
        'glob_norm_img_max': 0
    }
    tparams = {
        'input_normalize': 0,
        'input_height': 0,
        'input_width': 0,
        'color': color
    }
    _paths = list(paths)

    for i in range(len(paths)):

        [sample_img, sample_dsp] = loadSample(paths[i][0], paths[i][1], tparams)

        # Only consider non-outliers
        if np.min(sample_dsp) >= 0 and np.max(sample_dsp) <= sample_img.shape[1]:
            _params['glob_norm_dsp_min'] = min(_params['glob_norm_dsp_min'], np.min(sample_dsp))
            _params['glob_norm_dsp_max'] = max(_params['glob_norm_dsp_max'], np.max(sample_dsp))
            _params['glob_norm_img_min'] = min(_params['glob_norm_img_min'], np.min(sample_img))
            _params['glob_norm_img_max'] = max(_params['glob_norm_img_max'], np.max(sample_img))
        else:
            # Remove outliers
            del _paths[i]

    return [_paths, _params]

def miniBatch_gf(train_paths, params):

    i = 0
    epoch = 0
    batch = 0
    numTrain = len(train_paths)

    while True:

        # Start again with next epoch, if all data has been used
        if (i + params['batchsize']) > numTrain:
            random.shuffle(train_paths)
            epoch += 1
            batch = 0
            i = 0

        # Generate next minibatch
        minibatch_img = []
        minibatch_dsp = []
        for j in range(i, i + params['batchsize']):
            [sample_img, sample_dsp] = loadSample(train_paths[j][0], train_paths[j][1], params)
            minibatch_img.append(sample_img)
            minibatch_dsp.append(sample_dsp)

        yield [epoch, batch, np.float32(minibatch_img), np.float32(minibatch_dsp)]

        i += params['batchsize']
        batch += 1

def miniBatchFromQueue_gf(batchqueue, batchsize, numTrain):

    i = 0
    epoch = 0
    batch = 0

    while True:

        # Start again with next epoch, if all data has been used
        if (i + batchsize) > numTrain:
            epoch += 1
            batch = 0
            i = 0

        # Generate next minibatch
        minibatch_img = []
        minibatch_dsp = []
        for j in range(batchsize):
            while batchqueue.getNumSamples() < batchsize:
                print(">>> Caching batch queue, waiting 1s... <<<")
                time.sleep(1)
            [sample_img, sample_dsp] = batchqueue.getNextSample()
            minibatch_img.append(sample_img)
            minibatch_dsp.append(sample_dsp)

        yield [epoch, batch, np.float32(minibatch_img), np.float32(minibatch_dsp)]

        i += batchsize
        batch += 1

class MiniBatchQueue(threading.Thread):

    def __init__(self, paths, params):
        threading.Thread.__init__(self)
        self.q = deque()
        self.params = params
        self.paths = paths
        self.setDaemon(True)
        self.start()

    def run(self):
        i = 0
        while True:
            while self.getNumSamples() < self.params['batchqueuesize']:
                if i >= len(self.paths):
                    i = 0
                    random.shuffle(self.paths)
                self.q.append(copy.deepcopy(loadSample(self.paths[i][0], self.paths[i][1], self.params)))
                i += 1

    def getNumSamples(self):
        return len(self.q)

    def getNextSample(self):
        return copy.deepcopy(self.q.popleft())
