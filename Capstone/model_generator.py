###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# March 16th, 2018
#
# This file defines the generator model as U-Net.
###############################################################################

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Activation, Concatenate
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal

def model_generator_UNet():

    layer_i = Input(shape=(256, 256, 3))

    # Encoder
    layer_e1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_i)

    layer_e2 = LeakyReLU(alpha=0.2)(layer_e1)
    layer_e2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_e2)
    layer_e2 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_e2)

    layer_e3 = LeakyReLU(alpha=0.2)(layer_e2)
    layer_e3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_e3)
    layer_e3 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_e3)

    layer_e4 = LeakyReLU(alpha=0.2)(layer_e3)
    layer_e4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_e4)
    layer_e4 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_e4)

    layer_e5 = LeakyReLU(alpha=0.2)(layer_e4)
    layer_e5 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_e5)
    layer_e5 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_e5)

    layer_e6 = LeakyReLU(alpha=0.2)(layer_e5)
    layer_e6 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_e6)
    layer_e6 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_e6)

    layer_e7 = LeakyReLU(alpha=0.2)(layer_e6)
    layer_e7 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_e7)
    layer_e7 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_e7)

    layer_e8 = LeakyReLU(alpha=0.2)(layer_e7)
    layer_e8 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_e8)

    # Decoder
    layer_d1 = Activation('relu')(layer_e8)
    layer_d1 = Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_d1)
    layer_d1 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_d1)
    layer_d1 = Dropout(0.5)(layer_d1)
    layer_d1 = Concatenate()([layer_d1, layer_e7]) # Residual connections

    layer_d2 = Activation('relu')(layer_d1)
    layer_d2 = Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_d2)
    layer_d2 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_d2)
    layer_d2 = Dropout(0.5)(layer_d2)
    layer_d2 = Concatenate()([layer_d2, layer_e6]) # Residual connections

    layer_d3 = Activation('relu')(layer_d2)
    layer_d3 = Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_d3)
    layer_d3 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_d3)
    layer_d3 = Dropout(0.5)(layer_d3)
    layer_d3 = Concatenate()([layer_d3, layer_e5]) # Residual connections

    layer_d4 = Activation('relu')(layer_d3)
    layer_d4 = Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_d4)
    layer_d4 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_d4)
    layer_d4 = Concatenate()([layer_d4, layer_e4]) # Residual connections

    layer_d5 = Activation('relu')(layer_d4)
    layer_d5 = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_d5)
    layer_d5 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_d5)
    layer_d5 = Concatenate()([layer_d5, layer_e3]) # Residual connections

    layer_d6 = Activation('relu')(layer_d5)
    layer_d6 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_d6)
    layer_d6 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_d6)
    layer_d6 = Concatenate()([layer_d6, layer_e2]) # Residual connections

    layer_d7 = Activation('relu')(layer_d6)
    layer_d7 = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_d7)
    layer_d7 = BatchNormalization(momentum=0.9, epsilon=1.01e-5, axis=-1, gamma_initializer=RandomNormal(1.,0.02))(layer_d7)
    layer_d7 = Concatenate()([layer_d7, layer_e1]) # Residual connections

    layer_d8 = Activation('relu')(layer_d7)
    layer_d8 = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(0, 0.02))(layer_d8)

    layer_o = Activation('tanh')(layer_d8)

    return Model(inputs=[layer_i], outputs=[layer_o])
