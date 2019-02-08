import keras
import tensorflow as tf
from keras.layers import *


def binary_activation(x):
    z = tf.where(x >= 0.5, x - x + 1.0, x)
    y = tf.where(z <= -0.5, z - z + 0, z + 0.5)
    return y


def build_model():

    input = Input(shape=(128, 128, 3))
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(input)
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)

    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    encoded = Conv2D(32, (4, 4), strides=(2, 2), activation=binary_activation, padding='same')(x)

    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(3, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)

    model = keras.models.Model(input, x)
    print(model.summary())
    return model
