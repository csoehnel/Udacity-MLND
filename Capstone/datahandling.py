import sys
import cv2
import glob
import time
import random
import threading
import numpy as np
from collections import deque
from PIL import Image
from python_pfm import *

####################### DEBUG #######################
from random import randint

loadSize = 286
imageSize = 256

def read_image(fn):
    im = Image.open(fn)
    im = im.resize( (loadSize*2, loadSize), Image.BILINEAR )
    arr = np.array(im)/255*2-1
    w1,w2 = (loadSize-imageSize)//2,(loadSize+imageSize)//2
    h1,h2 = w1,w2
    imgA = arr[h1:h2, loadSize+w1:loadSize+w2, :]
    imgB = arr[h1:h2, w1:w2, :]
    if randint(0,1):
        imgA=imgA[:,::-1]
        imgB=imgB[:,::-1]
    return imgA, imgB

######################################################

def loadSample(img_path, dsp_path, color = False, width = 0, height = 0):

    sample_img = cv2.imread(img_path)
    [sample_dsp, scale] = readPFM(dsp_path)

    # Color or Grey-Scale
    if color:
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2RGB)
        sample_dsp = np.dstack([sample_dsp] * 3)
    else:
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2GRAY)

    # Resizing
    if width >= 1 and height >= 1:
        sample_img = cv2.resize(sample_img, (width, height), interpolation = cv2.INTER_AREA).astype('float32') # INTER_AREA best for shrinkage
        sample_dsp = cv2.resize(sample_dsp, (width, height), interpolation = cv2.INTER_AREA).astype('float32') # INTER_AREA best for shrinkage
    else:
         sample_img = sample_img.astype('float32')
         sample_dsp = sample_dsp.astype('float32')

    return [sample_img, sample_dsp]

def showSampleImage1(img_path, dsp_path, img_path_out, dsp_path_out):

    img = Image.open(img_path).convert("RGB")
    img.show()
    img.save(img_path_out)

    [data, scale] = readPFM(dsp_path)
    data = (data * 255 / np.max(data)).astype('uint8') # only for saving to jpg
    disparity_image = Image.fromarray(data)
    disparity_image.show()
    disparity_image.save(dsp_path_out)

def showSampleImage2(img_path, dsp_path):

    [sample_img, sample_dsp] = loadSample(img_path, dsp_path, True)
    sample_img = (sample_img * 255 / np.max(sample_img)).astype('uint8')
    sample_dsp = (sample_dsp * 255 / np.max(sample_dsp)).astype('uint8')
    Image.fromarray(sample_img).show()
    Image.fromarray(sample_dsp).show()

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

def miniBatch_gf(train_paths, batchsize, color = False, width = 0, height = 0):

    i = 0
    epoch = 0
    batch = 0
    numTrain = len(train_paths)

    while True:

    # Start again with next epoch, if all data has been used
        if (i + batchsize) > numTrain:
            random.shuffle(train_paths)
            epoch += 1
            batch = 0
            i = 0

        # Generate next minibatch
        minibatch_img = []
        minibatch_dsp = []
        for j in range(i, i + batchsize):
            #[sample_img, sample_dsp] = loadSample(train_paths[j][0], train_paths[j][1], color, width, height)
            [sample_img, sample_dsp] = read_image(train_paths[j][0])
            minibatch_img.append(sample_img)
            minibatch_dsp.append(sample_dsp)

        yield [epoch, batch, np.float32(minibatch_img), np.float32(minibatch_dsp)]

        i += batchsize
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
        for j in range(i, i + batchsize):
            while batchqueue.getNumSamples() < 1:
                print(">>> Batch queue empty, waiting 1s. <<<")
                time.sleep(1)
            [sample_img, sample_dsp] = batchqueue.getNextSample()
            minibatch_img.append(sample_img)
            minibatch_dsp.append(sample_dsp)

        yield [epoch, batch, np.float32(minibatch_img), np.float32(minibatch_dsp)]

        i += batchsize
        batch += 1

class MiniBatchQueue(threading.Thread):

    def __init__(self, memlimit, paths, color, width, height):
        threading.Thread.__init__(self)
        self.q = deque()
        self.memlimit = memlimit
        self.paths = paths
        self.color = color
        self.width = width
        self.height = height
        self.setDaemon(True)
        self.start()

    def run(self):
        while True:
            i = 0
            while sys.getsizeof(self.q) < self.memlimit:
                if i >= len(self.paths):
                    random.shuffle(self.paths)
                    i = 0
                self.q.append(loadSample(self.paths[i][0], self.paths[i][1], self.color, self.width, self.height))
                i += 1

    def getNumSamples(self):
        return len(self.q)

    def getNextSample(self):
        return self.q.popleft()
