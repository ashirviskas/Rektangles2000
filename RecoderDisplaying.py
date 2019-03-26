import NetworkBuilder as nb
import ImageReader as ir
import pygame
import keras
import numpy as np
import os
from keras.models import load_model
import time
import cv2
import tensorflow as tf
import png

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def binary_activation(x):
    return nb.binary_activation(x)


def fix_image_for_showing(img_, limits=(460, 460)):
    max_ind = np.argmax(np.array(img_.shape))
    scale_factor = limits[max_ind] / img_.shape[max_ind]
    if scale_factor > 1:
        scale_factor = int(scale_factor)
        img = img_.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)
    else:
        img = cv2.resize(img_, dsize=(int(scale_factor * img_.shape[1]), int(scale_factor * img_.shape[0])), interpolation=cv2.INTER_CUBIC)
    img = np.rot90(img, 1)
    Z = 255 * img / img.max()
    Z = np.array(Z, dtype=np.uint8)
    # c_d = cv2.cvtColor(Z, cv2.COLOR_BGR2RGB)
    c_d = cv2.cvtColor(Z, cv2.COLOR_HSV2RGB)
    surf = pygame.surfarray.make_surface(c_d)
    return surf


def fix_weight_size(weight, wanted_size=(64, 64)):
    max_ind = np.argmax(np.array(weight.shape))
    scale_factor = wanted_size[max_ind] / weight.shape[max_ind]
    if scale_factor > 1:
        scale_factor = int(scale_factor)
        w = weight.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)
        return w
    else:
        w = cv2.resize(weight, dsize=(int(scale_factor * weight.shape[1]), int(scale_factor * weight.shape[0])), interpolation=cv2.INTER_NEAREST)
        return w


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))


def generator_images(images):
    for im in images:
        im_expanded = np.expand_dims(im, axis=0)
        print("Doin Image")
        yield im_expanded


def visualise_model(images, model, layer_of_activations, layer_x=2, layer_y=5):
    fp_test = os.path.expanduser('~') + "/Bakk/Bakalauras/compression_testing/recodedImages/"
    encoded_model = keras.models.Model(inputs=model.input,
                                       outputs=[model.get_layer(layer_of_activations).get_output_at(0), model.output])
    images_recoded = list()
    # for im in images:
    #     strt = time.time()
    #     im_expanded = np.expand_dims(im, axis=0)
    #     print("Expanding: ", (time.time() - strt))
    #     strt = time.time()
    #     images_recoded.append(encoded_model.predict(im_expanded))
    #     print("Predicting: ", (time.time() - strt))
    images_recoded = encoded_model.predict_generator(generator=generator_images(images), steps=len(images))
    pygame.init()
    w = 1432
    h = 650
    size = (w, h)
    screen = pygame.display.set_mode(size)
    for m, im in enumerate(images):
        weights_shp = None
        image_weights_arr = np.array(images_recoded[0][m][0, :, :, :], dtype=np.bool)
        image_weights_arr = np.swapaxes(image_weights_arr, 0, 2)
        # image_weights_arr = np.swapaxes(image_weights_arr, 0, 2)
        image_weights_arr = np.rot90(image_weights_arr, 3, axes=(1, 2))
        iw_shp = image_weights_arr.shape
        image_weights_image = unblockshaped(image_weights_arr, iw_shp[1] * 8, iw_shp[0] * iw_shp[2] // 8)#np.reshape(image_weights_arr, newshape=image_weights_image.shape)
        writer = png.Writer(image_weights_image.shape[1], image_weights_image.shape[0], bitdepth=1,
                            greyscale=True, compression=9, planes=1)

        # file_write = open("./output_test/weights_" + str(m) + ".png", mode="wb")
        # file_write_array = open("./output_test/write_array.png", mode="wb")
        # file_write_packed = open("./output_test/write_packed.png", mode="wb")
        image_object = png.from_array(image_weights_image, "L").save("./output_test/weights_" + str(m) + ".png")
        # writer.write(file_write, image_weights_image)
        # writer.write_array(file_write_array, image_weights_image)
        # writer.write_packed(file_write_packed, image_weights_image)
        for i in range(layer_x):
            for j in range(layer_y):
                img = np.rot90(images_recoded[0][m][0, :, :, i * layer_y + j], )
                weights_shp = img.shape
                img = fix_weight_size(img)
                # img = img.repeat(2, axis=0).repeat(2, axis=1)
                Z = 255 * img / img.max()
                surf = pygame.surfarray.make_surface(Z)
                screen.blit(surf, (i * 66 + 460, j * 66))

        Z = 255 * images_recoded[1][m][0] / images_recoded[1][m][0].max()
        Z = np.array(Z, dtype=np.uint8)
        cv2.imwrite(fp_test + str(m) + ".jpg", Z)
        img_recoded = fix_image_for_showing(images_recoded[1][m][0, :, :, :])
        screen.blit(img_recoded, (972, 0))
        img_orig = fix_image_for_showing(im)
        screen.blit(img_orig, (0, 0))
        # weights_arr = im[0][0]
        # weights_arr = np.array(weights_arr, dtype=np.bool)
        weights_arr_binary_string = np.packbits(image_weights_arr).tobytes()
        # weights_arr.tofile("./encoded_images/64_16_16_png/" + str(m) + "_imgdat")
        with open("./encoded_images/64_16_16_png_2/" + str(m) + "_imgdatp", 'wb') as datafile:
            datafile.write(weights_arr_binary_string)
        myfont = pygame.font.SysFont('monospace', 30)
        img_txt = myfont.render("Image dimensions: " + str(im.shape[1]) + "x" + str(im.shape[0]), 1, (255, 255, 255))
        weight_txt = myfont.render("Compressed dimensions: " + str(weights_shp[1]) + "x" + str(weights_shp[0]) + "x64", 1, (255, 255, 255))
        screen.blit(img_txt, (0, 560))
        screen.blit(weight_txt, (0, 590))

        pygame.display.flip()
        time.sleep(0.1)
        screen.fill((0, 0, 0))



def main(modelname = "64i_15k_50b_330e_mixed_training"):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    fp = os.path.expanduser('~') + "/Downloads/img_celeba/data_crop_256_png"
    # fp = os.path.expanduser('~') + "/Bakk/Bakalauras/personal_testing_images/logos"
    # fp = os.path.expanduser('~') + "/Bakk/Bakalauras/personal_testing_images/cropped"
    # fp = os.path.expanduser('~') + "/Bakk/Bakalauras/personal_testing_images/random"
    images_n = 99
    images = ir.read_directory(fp, images_n, start=0)
    images = np.array(images) / 255

    model = load_model(modelname, custom_objects={'binary_activation': nb.binary_activation}, compile=True)
    print(model.summary())
    visualise_model(images, model, "conv2d_14", 8, 8)


if __name__ == "__main__":
    main()
