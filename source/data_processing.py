# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Class to process all datas -*-

# import libraries
import numpy as np
from sklearn.model_selection import train_test_split

# import files
from constants import *
from helpers import *

class ProcessingData:
    """Class to process all datas."""

    def __init__(self, imgs, gt_imgs):
        """Constructor.
        Args:
            imgs (np.ndarray): Images.
            gt_imgs (np.ndarray): Groundtruth images.
        """
        self.imgs = np.array(imgs)
        self.gt_imgs = np.array(gt_imgs)
        self.imgs_patches = np.array([])
        self.gt_imgs_patches = np.array([])
        self.imgs_train = np.array([])
        self.gt_imgs_train = np.array([])
        self.imgs_test = np.array([])
        self.gt_imgs_test = np.array([])
        self.imgs_validation = np.array([])
        self.gt_imgs_validation = np.array([])

    
    def create_patches(self, patch_size=PATCH_SIZE):
        """Create patches from the images.
        Args:
            PATCH_SIZE (int): Size of the patch.
        """
        print("Creating patches...")
        img_patches = [img_crop(self.imgs[i], patch_size, patch_size) for i in range(len(self.imgs))]
        gt_patches = [img_crop(self.gt_imgs[i], patch_size, patch_size) for i in range(len(self.imgs))]
        # Linearize list of patches
        self.imgs_patches = np.asarray(
            [
                img_patches[i][j]
                for i in range(len(img_patches))
                for j in range(len(img_patches[i]))
            ]
        )
        self.gt_imgs_patches = np.asarray(
            [
                gt_patches[i][j]
                for i in range(len(gt_patches))
                for j in range(len(gt_patches[i]))
            ]
        )
        print("Done!")
    

    def create_labels(self, threshold=FOREGROUND_THRESHOLD):
        """Create labels from the patches.
        Args:
            FOREGROUND_THRESHOLD (float): Threshold to determine if a patch is foreground or background.
        """
        print("Creating labels...")
        self.gt_imgs_patches = np.asarray(
            [
                1 if np.mean(self.gt_imgs_patches[i]) > threshold else 0
                for i in range(len(self.gt_imgs_patches))
            ]
        )
        print("Done!")


    def create_sets(self, validation_size=VALIDATION_SIZE, test_size=TEST_SIZE):
        """Split the data into train, test and validation sets."""
        print("Splitting data...")
        tmp_x, self.imgs_validation, tmp_y, self.gt_imgs_validation  = train_test_split(
            self.imgs_patches, self.gt_imgs_patches, test_size=validation_size, random_state=42
        )
        self.imgs_train, self.imgs_test, self.gt_imgs_train, self.gt_imgs_test = train_test_split(
            tmp_x, tmp_y, test_size=test_size, random_state=42
        )
        print("Done!")