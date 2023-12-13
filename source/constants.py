# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-12-13 (Yannis Laaroussi) -*-
# -*- python version : 3.11.6 -*-
# -*- Regroup all the constants used -*-

#import files
import os
import torch

#File paths
ROOT_DIR = "../data/"
TRAIN_DIR = os.path.join(ROOT_DIR, "training/")
TEST_DIR = os.path.join(ROOT_DIR, "test_set_images/")
IMAGE_DIR = os.path.join(TRAIN_DIR, "images/")
GT_DIR = os.path.join(TRAIN_DIR, "groundtruth/")
MODELS_DIR = "../models/"   
RESULTS_FOLDER_PATH = os.path.join(ROOT_DIR, "submission/images/")
SUBMISSION_PATH = os.path.join(ROOT_DIR, "submission/")

#Image parameters
PATCH_SIZE = 16
NUM_CHANNELS = 3

#Training parameters
NB_IMAGES_MAX = len(os.listdir(IMAGE_DIR))
NB_IMAGES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOREGROUND_THRESHOLD = 0.25
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
AUG_PATCH_SIZE = 16
SIGMA = 2
#DATA SIZE
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.3
TRAIN_SAMPLES= 128000
BATCH_SIZE = 32
TEST_SAMPLES = int(TRAIN_SAMPLES * TEST_RATIO)
VALIDATION_SAMPLES = int(TRAIN_SAMPLES * VALIDATION_RATIO)
