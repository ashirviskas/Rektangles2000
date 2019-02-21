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
    # server = tf.train.Server.create_local_server()
    # sess = tf.Session(server.target)
    # K.set_session(sess)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # K.set_floatx('float16')

    learning_rate = 0.0001
    epochs = 5
    batch_size = 20
    decay_r = (learning_rate / (epochs))
    images_n = 15000

    # model = nb.build_model()
    model = keras.models.load_model("64i_15k_50b_330e_png_z", custom_objects={'binary_activation': nb.binary_activation})
    fp = os.path.expanduser('~') + "/Downloads/img_celeba/data_crop_256_jpg"
    images = ir.read_directory(fp, images_n)
    images = np.array(images) / 255
    images = np.array(images, dtype=np.float16)
    loss_func = "mse"
    if True: # todo: implement multigpu setting
        multi_model = keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=True, cpu_relocation=False)
    else:
        multi_model = model
    multi_model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=decay_r), loss=loss_func,
                  metrics=['accuracy'])
    history = multi_model.fit(images, images, validation_split=0.05, callbacks=[], batch_size=batch_size, epochs=epochs)
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
    train("64i_15k_50b_330e_mixed_training")


if __name__ == "__main__":
    main()
