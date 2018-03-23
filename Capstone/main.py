###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# December 1st, 2017
#
# This is the main file of the algorithm.
###############################################################################

import os
import argparse
import tensorflow
#from keras.utils.training_utils import multi_gpu_model

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

# Load and prepare training data

# Generator model
modelG = model_generator_UNet()
modelG.summary()

# Discriminator model
modelD = model_discriminator(3)
modelD.summary()
