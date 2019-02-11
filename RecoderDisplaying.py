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


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    fp = os.path.expanduser('~') + "/Downloads/img_celeba/data_crop_128_jpg"
    images_n = 500
    images = ir.read_directory(fp, images_n)
    images = np.array(images) / 255

    model = load_model("128i_10k_64b_1e", custom_objects={'binary_activation':binary_activation})
    encoded_model = keras.models.Model(inputs = model.input, outputs=[model.get_layer("conv2d_7").get_output_at(0), model.output])
    images_recoded = encoded_model.predict(images)
    pygame.init()
    w = 1200
    h = 650
    size = (w, h)
    screen = pygame.display.set_mode(size)
    for m in range(images_n):
        for i in range(8):
            for j in range(16):
                img = images_recoded[0][m, :, :, i * 16 + j]
                img = img.repeat(4,axis=0).repeat(4,axis=1)
                Z = 255 * img / img.max()
                surf = pygame.surfarray.make_surface(Z)
                screen.blit(surf, (i * 5 * 8 + 460, j*5 * 8))

        img_recoded = fix_image_for_showing(images_recoded[1][m, :, :, :])
        screen.blit(img_recoded, (780, 0))
        img_orig = fix_image_for_showing(images[m, :, :, :])
        screen.blit(img_orig, (0, 0))
        weights_arr = images_recoded[0][m]
        weights_arr.tofile("./encoded_images/" + str(m) + "lol")


        pygame.display.flip()
        time.sleep(1)



if __name__ == "__main__":
    main()