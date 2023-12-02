# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-25 -*-
# -*- python version : 3.11.6 -*-
# -*- Functions to load the datas -*-

# import libraries
import os 
import numpy as np

#import files
from constants import*
from helpers import*

def load_datas(nb_img):
    """Load train and test sets from disk.
    Args:
        infilename (str): Path to the image file.
    Returns:
        np.ndarray: The loaded image.
    """
    files = os.listdir(IMAGE_DIR)
    n = min(nb_img, len(files))
    print("Loading " + str(n) + " images")
    imgs = [load_image(IMAGE_DIR + files[i]) for i in range(n)]
    print(files[0])

    print("Loading " + str(n) + " images")
    gt_imgs = [load_image(GT_DIR + files[i]) for i in range(n)]
    print(files[0])
    return np.asarray(imgs), np.asarray(gt_imgs)

def load_test_datas():
    """Load train and test sets from disk.
    Args:
        infilename (str): Path to the image file.
    Returns:
        np.ndarray: The loaded images.
    """
    files = os.listdir(TEST_DIR)
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[1]))
    imgs = [load_image(TEST_DIR + file + '/' + file + '.png') for file in sorted_files if not file.startswith('.')]
    return np.asarray(imgs)
