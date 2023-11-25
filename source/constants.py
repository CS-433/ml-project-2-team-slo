# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Regroup all the constants used -*-

#import files
import os

#File paths
ROOT_DIR = "../data/training/"
IMAGE_DIR = ROOT_DIR + "images/"
GT_DIR = ROOT_DIR + "groundtruth/"

NB_IMAGES_MAX = os.listdir(IMAGE_DIR)
NB_IMAGES = 10

#Image size
PATCH_SIZE = 16

#Training parameters
FOREGROUND_THRESHOLD = 0.25
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.3