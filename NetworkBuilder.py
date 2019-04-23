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


def add_residual(inp, filters_n):
    out = Conv2D(filters_n, (3, 3), strides=(1, 1), activation='relu', padding='same')(inp)
    out = Conv2D(int(inp.shape[3]), (3, 3), strides=(1, 1), activation='relu', padding='same')(out)
    out = keras.layers.add([inp, out])
    return out


def build_model():
    input = Input(shape=(None, None, 3))
    x = Conv2D(256, (8, 8), strides=(4, 4), activation='relu', padding='same')(input)
    x = add_residual(x, 128)
    x = Conv2D(256, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    x = add_residual(x, 256)
    x = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = add_residual(x, 256)
    x = add_residual(x, 256)
    x = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), activation='sigmoid', padding='same')(x)
    encoded = Conv2D(64, (1, 1), strides=(1, 1), activation=binary_activation, padding='same')(x)

    # x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(396, (1, 1), strides=(1, 1), activation='relu', padding='same')(encoded)
    x = add_residual(x, 396)
    x = add_residual(x, 396)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = add_residual(x, 128)
    x = UpSampling2D((4, 4))(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(128, (6, 6), strides=(1, 1), activation='relu', padding='same')(x)
    x = add_residual(x, 128)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = add_residual(x, 128)
    # x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(3, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    model = keras.models.Model(input, x)
    print(model.summary(200))
    return model


if __name__ == '__main__':
    build_model()

