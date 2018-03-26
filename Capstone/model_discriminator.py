###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# March 16th, 2018
#
# This file defines the discriminator model.
# Implementation inspired by:
# https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix
# https://github.com/tjwei/GANotebooks/blob/master/pix2pix-keras.ipynb
# https://github.com/phillipi/pix2pix
###############################################################################

from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, BatchNormalization, Activation
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal

def model_discriminator_pixel(params):

    channels = 3 if params['input_color'] else 1
    layer_i0 = Input(shape = (256, 256, channels))
    layer_i1 = Input(shape = (256, 256, channels))
    layer_i = Concatenate(axis = -1)([layer_i0, layer_i1])
    layer_h = Conv2D(64, (1, 1), strides = (1, 1), padding = 'same', kernel_initializer = RandomNormal(0, 0.02))(layer_i)
    layer_h = LeakyReLU(alpha = 0.2)(layer_h)
    layer_h = Conv2D(128, (1, 1), strides = (1, 1), padding = 'same', kernel_initializer = RandomNormal(0, 0.02))(layer_h)
    layer_h = BatchNormalization(momentum = 0.9, epsilon = 1.01e-5, axis = -1, gamma_initializer = RandomNormal(1., 0.02))(layer_h)
    layer_h = LeakyReLU(alpha = 0.2)(layer_h)
    layer_o = Conv2D(1, (1, 1), strides = (1, 1), padding = 'same', kernel_initializer = RandomNormal(0, 0.02))(layer_h)
    layer_o = Activation('sigmoid')(layer_o)

    return Model(inputs=[layer_i0, layer_i1], outputs=[layer_o])

def model_discriminator(num_layers, params):

    # Use pixel GAN?
    if num_layers == 0:
        return model_discriminator_pixel()

    channels = 3 if params['input_color'] else 1
    layer_i0 = Input(shape = (256, 256, channels))
    layer_i1 = Input(shape = (256, 256, channels))
    layer_i = Concatenate(axis = -1)([layer_i0, layer_i1])
    layer_h = Conv2D(64, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = RandomNormal(0, 0.02))(layer_i)
    layer_h = LeakyReLU(alpha = 0.2)(layer_h)

    for n in range(1, num_layers):
        layer_h = Conv2D(64 * min(2**n, 8), (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = RandomNormal(0, 0.02))(layer_h)
        layer_h = BatchNormalization(momentum = 0.9, epsilon = 1.01e-5, axis = -1, gamma_initializer = RandomNormal(1., 0.02))(layer_h)
        layer_h = LeakyReLU(alpha = 0.2)(layer_h)

    layer_h = Conv2D(64 * min(2**num_layers, 8), (4, 4), strides = (1, 1), padding = 'same', kernel_initializer = RandomNormal(0, 0.02))(layer_h)
    layer_h = BatchNormalization(momentum = 0.9, epsilon = 1.01e-5, axis = -1, gamma_initializer = RandomNormal(1., 0.02))(layer_h)
    layer_h = LeakyReLU(alpha = 0.2)(layer_h)
    layer_o = Conv2D(1, (4, 4), strides = (1, 1), padding = 'same', kernel_initializer = RandomNormal(0, 0.02))(layer_h)
    layer_o = Activation('sigmoid')(layer_o)

    return Model(inputs=[layer_i0, layer_i1], outputs=[layer_o])
