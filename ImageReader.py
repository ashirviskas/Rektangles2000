import os
import cv2
import numpy as np
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_all_filepaths_in_dir(directory):
    filepaths = os.listdir(directory)
    return filepaths


def read_directory(directory, limit=10000, start=0):
    filepaths = get_all_filepaths_in_dir(directory)
    images = list()
    logging.info(len(filepaths))
    for i, fp in enumerate(filepaths[start: start + limit]):
        images.append(cv2.imread(directory + "/" + fp, 3))
    return images







