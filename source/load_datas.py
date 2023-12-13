# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-12-13 (Vincent Roduit, Yannis Laaroussi) -*-
# -*- python version : 3.11.6 -*-
# -*- Functions to load the datas -*-

# import libraries
import os
import numpy as np

# import files
from constants import *
from helpers import *


def load_datas(nb_img, train_path=TRAIN_DIR):
    """Load train and test sets from disk.
    Args:
        infilename (str): Path to the image file.
    Returns:
        np.ndarray: The loaded image.
    """
    img_dir = os.path.join(train_path, "images/")
    gt_dir = os.path.join(train_path, "groundtruth/")
    files = os.listdir(img_dir)
    files = sorted(files, reverse=False)
    n = min(nb_img, len(files))
    imgs = [load_image(os.path.join(img_dir, files[i])) for i in range(n)]
    gt_imgs = [load_image(os.path.join(gt_dir, files[i])) for i in range(n)]

    return np.asarray(imgs), np.asarray(gt_imgs)


def load_test_datas(test_path=TEST_DIR):
    """Load train and test sets from disk.
    Args:
        infilename (str): Path to the image file.
    Returns:
        np.ndarray: The loaded images.
    """
    files = os.listdir(test_path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    sorted_files = sorted(files, key=lambda x: int(x.split("_")[1]))
    imgs = [
        load_image(os.path.join(test_path, file, file + ".png"))
        for file in sorted_files
        if not file.startswith(".")
    ]
    return np.asarray(imgs)
