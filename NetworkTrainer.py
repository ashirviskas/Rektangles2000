from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import numpy as no
import keras
import tensorflow as tf
from keras.layers import *
import NetworkBuilder as nb
import ImageReader as ir
import cv2
import os
from matplotlib import pyplot as plt
import RecoderDisplaying as rd


def train(modelname):
    learning_rate = 0.0001
    epochs = 100
    batch_size = 64
    decay_r = (learning_rate / (epochs))
    images_n = 15000

    model = nb.build_model()
    fp = os.path.expanduser('~') + "/Downloads/img_celeba/data_crop_128_png"
    images = ir.read_directory(fp, images_n)
    images = np.array(images) / 255
    loss_func = "mse"
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=decay_r), loss=loss_func,
                  metrics=['accuracy'])
    history = model.fit(images, images, validation_split=0.05, callbacks=[], batch_size=batch_size, epochs=epochs)
    model.save(modelname)

    test_im = ir.read_directory(fp, limit=200, start=images_n - 50)
    test_im_disp = np.array(test_im)
    test_im = np.array(test_im) / 255
    test_pred = model.predict(test_im)
    test_pred = np.array(test_pred * 255, dtype=np.uint8)
    rd.main(modelname)
    for i, d in enumerate(test_pred):
        break
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        c_d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        ax0.imshow(c_d, interpolation='nearest', aspect='auto')
        c_test_im = cv2.cvtColor(test_im_disp[i], cv2.COLOR_BGR2RGB)
        ax1.imshow(c_test_im, interpolation='nearest', aspect='auto')
        plt.show()
        if i > 50:
            break


def main():
    train("128i_15k_64b_100e_png_z")


if __name__ == "__main__":
    main()
