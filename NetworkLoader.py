import NetworkBuilder as nb
import keras
from keras.models import load_model
import tensorflow as tf
from keras.layers import *

from keras import backend as K

class NetworkLoader:
    def __init__(self, model_path, multigpu=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        self.model = load_model(model_path, custom_objects={'binary_activation': nb.binary_activation}, compile=True)

    def get_encoder(self, layer_of_activations = "conv2d_14"):
        encoder_model = keras.models.Model(inputs=self.model.input,
                                           outputs=self.model.get_layer(layer_of_activations).get_output_at(0))
        return encoder_model

    def get_decoder(self, layer_of_activations="conv2d_15"):
        for i, l in enumerate(self.model.layers[0:19]):
            # print(l.name)
            # print(l.output_shape)
            self.model.layers.pop(0)
            # print(self.model.summary())
        print(K.image_data_format())
        print(self.model.summary)
        newInput = Input(batch_shape=(None, None, None, 64))  # let us say this new InputLayer
        newOutputs = self.model(newInput)
        print(self.model.summary)
        newModel = keras.models.Model(newInput, newOutputs)
        print(newModel.summary(200))
        # decoder_model = keras.models.Model(inputs=self.model.get_layer(layer_of_activations).input,
        #                                    outputs=self.model.output)

        return newModel

