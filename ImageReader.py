import os
import cv2
import numpy as np
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_all_filepaths_in_dir(directory):
    filepaths = os.listdir(directory)
    filepaths = sorted(filepaths)
    return filepaths


def read_directory(directory, limit=10000, start=0):
    filepaths = get_all_filepaths_in_dir(directory)
    images = list()
    logging.info(len(filepaths))
    for i, fp in enumerate(filepaths[start: start + limit]):
        img = cv2.imread(directory + "/" + fp, 3)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cv2.imshow("img", img)
        # cv2.imshow("hsv", hsv)
        images.append(img)
        # k = cv2.waitKey(5) & 0xFF
        # if k == 27:
        #     break
    return images


def read_file(filepath):
    img = cv2.imread(filepath, 3)
    return img






