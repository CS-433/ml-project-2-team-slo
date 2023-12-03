# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Class to process all datas -*-

# import libraries
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import constants

# import files
from constants import *
from helpers import *

class BasicProcessing:
    """Class to process all datas. All the attributes are numpy arrays."""

    def __init__(self, imgs, gt_imgs = None):
        """Constructor.
        Args:
            imgs (np.ndarray): Images.
            gt_imgs (np.ndarray): Groundtruth images.
        """
        self.imgs = np.array(imgs)
        if gt_imgs is not None:
            self.gt_imgs = np.array(gt_imgs)
            self.imgs_patches = np.array([])
            self.gt_imgs_patches = np.array([])
            self.imgs_train = np.array([])
            self.gt_imgs_train = np.array([])
            self.imgs_test = np.array([])
            self.gt_imgs_test = np.array([])
            self.imgs_validation = np.array([])
            self.gt_imgs_validation = np.array([])
        else :
            self.gt_imgs = None

    
    def create_patches(self, patch_size=PATCH_SIZE):
        """Create patches from the images.
        Args:
            PATCH_SIZE (int): Size of the patch.
        """
        print("Creating patches...")
        img_patches = [img_crop(self.imgs[i], patch_size, patch_size) for i in range(len(self.imgs))]
        # Linearize list of patches

        self.imgs_patches = np.asarray(
            [
                img_patches[i][j]
                for i in range(len(img_patches))
                for j in range(len(img_patches[i]))
            ]
        )
        if self.gt_imgs is not None: 
            gt_patches = [img_crop(self.gt_imgs[i], patch_size, patch_size) for i in range(len(self.imgs))]
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
        """Split the data into train, test and validation sets.
        Args:
            VALIDATION_SIZE (float): Size of the validation set.
            TEST_SIZE (float): Size of the test set.
        """
        print("Splitting data...")
        tmp_x, self.imgs_test, tmp_y, self.gt_imgs_test  = train_test_split(
            self.imgs_patches, self.gt_imgs_patches, test_size=test_size, random_state=42
        )
        self.imgs_train, self.imgs_validation, self.gt_imgs_train, self.gt_imgs_validation = train_test_split(
            tmp_x, tmp_y, test_size=validation_size, random_state=42
        )
        print("Done!")
    def permute_axis(self):
        """Permute the axis of the images."""
        print("Permuting axis...")
        self.imgs_patches = np.transpose(self.imgs_patches, (0, 3, 1, 2))
        print("Done!")
    
    def proceed(self):
        """Proceed to the basic processing."""
        self.create_patches()
        self.create_labels()
        self.permute_axis()
        self.create_sets()

class AdvancedProcessing:
    """ Class to process all datas in a more advanced way. All the attributes are numpy arrays."""
    def __init__(self, 
                 imgs, 
                 gt_imgs = None, 
                 patch_size=PATCH_SIZE,
                 threshold=FOREGROUND_THRESHOLD,
                 validation_size=VALIDATION_RATIO,
                 test_size=TEST_RATIO):
        """Constructor.
        Args:
            imgs (np.ndarray): Images.
            gt_imgs (np.ndarray): Groundtruth images.
        """
        self.imgs = np.array(imgs)
        self.patch_size = patch_size
        self.threshold = threshold
        self.validation_size = validation_size
        self.test_size = test_size
        if gt_imgs is not None:
            self.gt_imgs = np.array(gt_imgs)
            self.imgs_patches = np.array([])
            self.gt_imgs_patches = np.array([])
            self.imgs_train = np.array([])
            self.gt_imgs_train = np.array([])
            self.imgs_test = np.array([])
            self.gt_imgs_test = np.array([])
            self.imgs_validation = np.array([])
            self.gt_imgs_validation = np.array([])
        else :
            self.gt_imgs = None
    
    def standardize_color(self):
        """Standardize the images."""
        print("Standardizing...")
        means = np.mean(self.imgs, axis=(0, 1, 2))
        stds = np.std(self.imgs, axis=(0, 1, 2))
        for i in range(NUM_CHANNELS):
            self.imgs[:, :, :, i] = (self.imgs[:, :, :, i] - means[i]) / stds[i]
        print("Done!")
    
    def split_sets(self):
        """Split the data into train, test and validation sets."""
        print("Splitting data...")
        tmp_x, self.imgs_test, tmp_y, self.gt_imgs_test  = train_test_split(
            self.imgs, self.gt_imgs, test_size=self.test_size, random_state=42
        )
        self.imgs_train, self.imgs_validation, self.gt_imgs_train, self.gt_imgs_validation = train_test_split(
            tmp_x, tmp_y, test_size=self.validation_size, random_state=42
        )
        print("Done!")
    
    def create_patches(self):
        """Create patches from the images."""
        print("Creating patches...")

        print("Done!")
    
    def proceed(self, train_size=TRAIN_SAMPLES,test_size=TEST_SAMPLES, validation_size=VALIDATION_SAMPLES):
        """Proceed to the advanced processing."""
        self.standardize_color()
        self.split_sets()
        self.create_patches()
    

    