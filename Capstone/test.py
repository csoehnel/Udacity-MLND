###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# March 16th, 2018
#
# This is the main file for testing.
###############################################################################

import os
import time
import numpy as np
from keras import backend as K
from keras.models import load_model
from datahandling import *
from evaluation import *
from vislog import *

### Params ###
params = {
    '_comment': '',
    'input_color': True,
    'input_height': 256,
    'input_normalize': 1,
    'input_width': 256,
    'path_generator': './logs/2018-MM-DD_hh-mm-ss/generator.h5',
    'pathpattern_img': '/home/XX/FlyingThings3D/frames_cleanpass_webp/TEST/**/left/*.webp',
    'pathpattern_dsp': '/home/XX/FlyingThings3D/disparity/TEST/**/left/*.pfm',
}

# Save params to file
outfolder = params['path_generator'].split('/generator')[0]
with open(os.path.join(outfolder, "params_test.log"), "w") as paramfile:
    paramfile.write(repr(params))

# Clone stdout to file
sys.stdout = Logger(sys.stdout, os.path.join(outfolder, "console_test.log"))

# Configure KERAS to use TENSORFLOW as backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Set to test mode ???
K.set_learning_phase(0)

# Load model
modelG = load_model(params['path_generator'])

# Set to test mode ???
K.set_learning_phase(0)

# 'Real' mono image is input tensor of generator
real_img = modelG.input

# 'Fake' disparity image is output tensor of generator
fake_dsp = modelG.output

# Function for ACTUAL RUN of generator (evaluation)
fn_runG = K.function([real_img], [fake_dsp])

### Prepare and load data ###

# Load and prepare test data
print("Preparing data...", end = " ")
t_data_start = time.time()
[test_paths, num_test] = loadFilePaths(params['pathpattern_img'], params['pathpattern_dsp'], True)
print("done after {:4.2f}s. Found {:d} test samples.".format(time.time() - t_data_start, num_test))

# Preloading test dataset
print("Fetching test data...", end = " ")
t_data_start = time.time()
[test_img, test_dsp] = loadImageSet(test_paths, 0, num_test, params['input_color'], params['input_width'],
                                      params['input_height'], params['input_normalize'])
print("done after {:4.2f}s.".format(time.time() - t_data_start))

### Evaluation ###

print("Evaluating {:d} test samples...".format(int(num_test)), end = " ")
t_eval_start = time.time()
epe = evaluate(test_img, test_dsp, fn_runG)
print("done after {:8.2f}s ==>> EPE = {:9.5f}".format(time.time() - t_eval_start, epe))
