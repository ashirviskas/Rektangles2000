import NetworkBuilder as nb
import ImageReader as ir
import pygame
import keras
import numpy as np
import os
from keras.models import load_model
import time
import cv2


def binary_activation(x):
    return nb.binary_activation(x)


def fix_image_for_showing(img_):
    img = img_.repeat(3, axis=0).repeat(3, axis=1)
    img = np.rot90(img, 1)
    Z = 255 * img / img.max()
    Z = np.array(Z, dtype=np.uint8)
    c_d = cv2.cvtColor(Z, cv2.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(c_d)
    return surf


def visualise_model(images, model, layer_of_activations, layer_x=2, layer_y=5):

    encoded_model = keras.models.Model(inputs=model.input,
                                       outputs=[model.get_layer(layer_of_activations).get_output_at(0), model.output])
    images_recoded = encoded_model.predict(images)
    pygame.init()
    w = 1200
    h = 650
    size = (w, h)
    screen = pygame.display.set_mode(size)
    for m, im in enumerate(images):
        for i in range(layer_x):
            for j in range(layer_y):
                img = np.rot90(images_recoded[0][m, :, :, i * layer_y + j], )
                img = img.repeat(2, axis=0).repeat(2, axis=1)
                Z = 255 * img / img.max()
                surf = pygame.surfarray.make_surface(Z)
                screen.blit(surf, (i * 5 * 8 + 460, j * 5 * 8))

        img_recoded = fix_image_for_showing(images_recoded[1][m, :, :, :])
        screen.blit(img_recoded, (780, 0))
        img_orig = fix_image_for_showing(images[m, :, :, :])
        screen.blit(img_orig, (0, 0))
        weights_arr = images_recoded[0][m]
        weights_arr = np.array(weights_arr, dtype=np.bool)
        weights_arr_binary_string = np.packbits(weights_arr).tobytes()
        # weights_arr.tofile("./encoded_images/64_16_16_png/" + str(m) + "_imgdat")
        with open("./encoded_images/64_16_16_png_2/" + str(m) + "_imgdatp", 'wb') as datafile:
            datafile.write(weights_arr_binary_string)

        pygame.display.flip()
        time.sleep(1)


def main(modelname = "32i_15k_50b_120e_png_z3"):
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    fp = os.path.expanduser('~') + "/Downloads/img_celeba/data_crop_128_png"
    # fp = os.path.expanduser('~') + "/Bakk/Bakalauras/personal_testing_images/logos"
    images_n = 20
    images = ir.read_directory(fp, images_n, start=0)
    images = np.array(images) / 255

    model = load_model(modelname, custom_objects={'binary_activation': binary_activation})
    print(model.summary())
    visualise_model(images, model, "conv2d_14", 8, 16)



if __name__ == "__main__":
    main()
