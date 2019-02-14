import keras
import tensorflow as tf
from keras.layers import *
import keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops


def _to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.
  Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.
  Returns:
      A tensor.
  """
  return ops.convert_to_tensor(x, dtype=dtype)


@tf.custom_gradient
def binary_activation(x):
    # ones = tf.fill(tf.shape(x), 1.0 + 0.01 * tf.keras.backend.argmax(x, -1), dtype=x.dtype.base_dtype)
    # zeros = tf.fill(tf.shape(x), 0.1 - 0.01 * tf.keras.backend.argmax(x, -1), dtype=x.dtype.base_dtype)
    zeros = K.zeros(K.shape(x), dtype=x.dtype.base_dtype)
    ones = K.ones(K.shape(x), dtype=x.dtype.base_dtype)

    def grad(dy):
        # Todo: Where x > 1, gradient 0/inverted. Where x < 0, same/opposite. Or where "this and gradient opposite",
        #  grad 0, otherwise dy
        #
        #clipped = clip_ops.clip_by_value(x + dy, zeros, ones)
        return dy #* (1 - 1 / (1 + clipped))

    return keras.backend.switch(x > 0.5, ones, zeros), grad


def build_model():
    input = Input(shape=(128, 128, 3))
    x = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='same')(input)
    # x = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='sigmoid', padding='same')(x)
    encoded = Conv2D(128, (4, 4), strides=(2, 2), activation=binary_activation, padding='same')(x)

    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    # x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    # x = Conv2D(64, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(3, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)

    model = keras.models.Model(input, x)
    print(model.summary())
    return model


if __name__ == '__main__':
    build_model()

