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


def binary_activation(x):
    # ones = tf.fill(tf.shape(x), 1.0 + 0.01 * tf.keras.backend.argmax(x, -1), dtype=x.dtype.base_dtype)
    # zeros = tf.fill(tf.shape(x), 0.1 - 0.01 * tf.keras.backend.argmax(x, -1), dtype=x.dtype.base_dtype)
    zeros = tf.zeros(tf.shape(x), dtype=x.dtype.base_dtype)
    ones = tf.ones(tf.shape(x), dtype=x.dtype.base_dtype)
    zeros = zeros + x * 1e-10
    ones = ones + x * 1e-10


  # x = (x - 0.5) * 10e5
  # zero = _to_tensor(0., x.dtype.base_dtype)
  # one = _to_tensor(1., x.dtype.base_dtype)
  # x = clip_ops.clip_by_value(x, zero, one)
    return keras.backend.switch(x > 0.5, ones, zeros)

def build_model():

    input = Input(shape=(None, None, 3))
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(input)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='sigmoid', padding='same')(x)
    encoded = Conv2D(128, (4, 4), strides=(2, 2), activation=binary_activation, padding='same')(x)

    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(64, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(3, (4, 4), strides=(1, 1), activation='relu', padding='same')(x)

    model = keras.models.Model(input, x)
    print(model.summary())
    return model
