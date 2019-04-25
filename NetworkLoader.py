import NetworkBuilder as nb
import keras
from keras.models import load_model
import tensorflow as tf
from keras.layers import *
import NetworkBuilder as nb

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

        # Looping through the old model and popping the encoder part + encoded layer
        for i, l in enumerate(self.model.layers[0:19]):
            self.model.layers.pop(0)
            print(self.model.summary())

        # Building a clean model that is the exact same architecture as the decoder part of the autoencoder
        new_model = nb.build_decoder()

        # Looping through both models and setting the weights on the new decoder
        for i, l in enumerate(self.model.layers):
            print(i, l.name, l.output_shape)
            print(new_model.layers[i+1].name, new_model.layers[i+1].output_shape)
            new_model.layers[i+1].set_weights(l.get_weights())
        print(self.model.summary)
        print(new_model.summary(200))
        return new_model

