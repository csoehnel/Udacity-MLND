###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# December 1st, 2017
#
# This file defines the discriminator models.
###############################################################################

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import LeakyReLU

def model_discriminator_pixel():

    model = Sequential()
    model.add(Conv2D(64, (1, 1), strides = (1, 1), padding = 'same', input_shape = (256, 256, 6)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (1, 1), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(1, (1, 1), strides = (1, 1), padding = 'same'))
    model.add(Activation('sigmoid'))

    return model    

def model_discriminator(num_layers):

    # Use pixel GAN?
    if num_layers == 0:
        return model_discriminator_pixel()

    model = Sequential()
    model.add(Conv2D(64, (4, 4), strides = (2, 2), padding = 'same', input_shape = (256, 256, 6)))
    model.add(LeakyReLU(0.2))

    for n in range(1, num_layers):
        model.add(Conv2D(64 * min(2**n, 8), (4, 4), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

    model.add(Conv2D(64 * min(2**num_layers, 8), (4, 4), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(1, (4, 4), strides = (1, 1), padding = 'same'))
    model.add(Activation('sigmoid'))

    return model
