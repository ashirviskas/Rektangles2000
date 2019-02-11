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


def main():
    learning_rate = 0.0001
    epochs = 20
    batch_size = 64
    decay_r = 0 #(learning_rate / (epochs + epochs / 2))
    images_n = 10000

    model = nb.build_model()
    fp = os.path.expanduser('~') + "/Downloads/img_celeba/data_crop_128_jpg"
    images = ir.read_directory(fp, images_n)
    images = np.array(images) / 255
    loss_func = "mse"
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=decay_r), loss=loss_func, metrics=['accuracy'])
    history = model.fit(images, images, validation_split=0.05, callbacks=[], batch_size=batch_size, epochs=epochs)
    model.save("128i_10k_64b_20e")

    test_im = ir.read_directory(fp, limit=200, start=images_n-50)
    test_im_disp = np.array(test_im)
    test_im = np.array(test_im) / 255
    test_pred = model.predict(test_im)
    test_pred = np.array(test_pred * 255, dtype=np.uint8)

    for i, d in enumerate(test_pred):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        c_d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        ax0.imshow(c_d, interpolation='nearest', aspect='auto')
        c_test_im = cv2.cvtColor(test_im_disp[i], cv2.COLOR_BGR2RGB)
        ax1.imshow(c_test_im, interpolation='nearest', aspect='auto')
        plt.show()
        if i > 50:
            break

    for i in range(5):
        break
        cv2.imshow('image', images[i])
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()


if __name__ == "__main__":

    main()