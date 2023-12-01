# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Class to process all datas -*-

# import libraries
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# import files
from constants import *
from helpers import *

class ProcessingData:
    """Class to process all datas."""

    def __init__(self, imgs, gt_imgs = None):
        """Constructor.
        Args:
            imgs (np.ndarray): Images.
            gt_imgs (np.ndarray): Groundtruth images.
        """
        self.imgs = np.array(imgs)
        if gt_imgs :
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
        if self.gt_imgs : 
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
    def compute_mean_std(self):
        """Compute mean and standard deviation of the train set."""
        print("Computing mean and std...")
        mean_r = np.mean(self.imgs[:, 0, :, :])
        mean_g = np.mean(self.imgs[:, 1, :, :])
        mean_b = np.mean(self.imgs[:, 2, :, :])

        means = np.array([mean_r, mean_g, mean_b])

        std_r = np.std(self.imgs[:, 0, :, :])
        std_g = np.std(self.imgs[:, 1, :, :])
        std_b = np.std(self.imgs[:, 2, :, :])

        stds = np.array([std_r, std_g, std_b])
        
        return means, stds
    def permute_axis(self):
        """Permute the axis of the images."""
        print("Permuting axis...")
        self.imgs_patches = np.transpose(self.imgs_patches, (0, 3, 1, 2))
        print("Done!")
    
    def balance_dataset(self):
        """Balance the dataset."""
        print("Balancing dataset...")
        # Get indexes of the patches
        index_0 = np.where(self.gt_imgs_patches == 0)[0]
        index_1 = np.where(self.gt_imgs_patches == 1)[0]
        # Get the number of patches
        n_0 = len(index_0)
        n_1 = len(index_1)
        # Get the difference between the number of patches
        diff = n_0 - n_1
        # Get the indexes of the patches to remove
        index_to_remove = np.random.choice(index_0, diff, replace=False)
        # Remove the patches
        self.imgs_patches = np.delete(self.imgs_patches, index_to_remove, axis=0)
        self.gt_imgs_patches = np.delete(self.gt_imgs_patches, index_to_remove)
        print("Done!")