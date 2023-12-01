# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Regroup all the constants used -*-

#import files
import os

#File paths
ROOT_DIR = "../data/"
TRAIN_DIR = ROOT_DIR + "training/"
TEST_DIR = ROOT_DIR + "test_set_images/"
IMAGE_DIR = os.path.join(TRAIN_DIR, "images/")
GT_DIR = TRAIN_DIR + "groundtruth/"

NB_IMAGES_MAX = len(os.listdir(IMAGE_DIR))
NB_IMAGES = 10

#Image size
PATCH_SIZE = 16

#Training parameters
FOREGROUND_THRESHOLD = 0.3
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.3
