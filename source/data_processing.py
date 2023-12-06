# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-12-03 -*-
# -*- python version : 3.11.6 -*-
# -*- Class to process all datas -*-

# import libraries
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader

# import files
import constants
from helpers import *
from load_datas import load_datas
from data_augmentation import generate_samples, create_patches_test_set

class BasicProcessing:
    """Class to process all datas. All the attributes are numpy arrays."""

    def __init__(self):
        """Constructor.
        Args:
            imgs (np.ndarray): Images.
            gt_imgs (np.ndarray): Groundtruth images.
        """
        self.imgs = np.array([])
        self.gt_imgs = np.array([])
        self.imgs_patches = np.array([])
        self.gt_imgs_patches = np.array([])
        self.imgs_train = np.array([])
        self.gt_imgs_train = np.array([])
        self.imgs_test = np.array([])
        self.gt_imgs_test = np.array([])
        self.imgs_validation = np.array([])
        self.gt_imgs_validation = np.array([])

    
    def create_patches(self, patch_size=constants.PATCH_SIZE):
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
        gt_patches = [img_crop(self.gt_imgs[i], patch_size, patch_size) for i in range(len(self.gt_imgs))]
        self.gt_imgs_patches = np.asarray(
            [
                gt_patches[i][j]
                for i in range(len(gt_patches))
                for j in range(len(gt_patches[i]))
            ]
        )
        print("Done!")
    

    def create_labels(self, threshold=constants.FOREGROUND_THRESHOLD):
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


    def create_sets(self, validation_size=constants.VALIDATION_RATIO, test_size=constants.TEST_RATIO):
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
    def load_data(self):
        """Load the data."""
        print("Loading data...")
        self.imgs, self.gt_imgs = load_datas(nb_img=constants.NB_IMAGES)
        print("Done!")
    def proceed(self):
        """Proceed to the basic processing."""
        self.load_data()
        self.create_patches()
        self.create_labels()
        self.permute_axis()
        self.create_sets()

class AdvancedProcessing:
    """ Class to process all datas in a more advanced way."""
    def __init__(self, 
                 nb_images=constants.NB_IMAGES, 
                 patch_size=constants.PATCH_SIZE,
                 window_size=constants.WINDOW_SIZE,
                 threshold=constants.FOREGROUND_THRESHOLD,
                 validation_size=constants.VALIDATION_RATIO,
                 batchsize=constants.BATCH_SIZE,
                 num_workers=constants.NUM_WORKERS,
                 num_samples=constants.TRAIN_SAMPLES,
                 upsample=True,
                 standardize=True):
        """Constructor.
        Args:
            imgs (np.ndarray): Images.
            gt_imgs (np.ndarray): Groundtruth images.
        """
        self.nb_images = nb_images
        self.patch_size = patch_size
        self.window_size = window_size
        self.threshold = threshold
        self.validation_size = validation_size
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.upsample = upsample
        self.num_samples = num_samples
        self.standardize = standardize
        self.imgs = np.array([])
        self.imgs_train = np.array([])
        self.gt_imgs_train = np.array([])
        self.imgs_validation = np.array([])
        self.gt_imgs_validation = np.array([])
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_validation = np.array([])
        self.y_validation = np.array([])

    def load_data(self):
        """Load the data."""
        print("Loading data...")
        self.imgs, self.gt_imgs = load_datas(self.nb_images)
        print("Done!")
    
    def standardize_color(self):
        """Standardize the images."""
        print("Standardizing...")
        means = np.mean(self.imgs, axis=(0, 1, 2))
        stds = np.std(self.imgs, axis=(0, 1, 2))
        for i in range(constants.NUM_CHANNELS):
            self.imgs[:, :, :, i] = (self.imgs[:, :, :, i] - means[i]) / stds[i]
        print("Done!")
    
    def split_sets(self):
        """Split the data into train, test and validation sets."""
        print("Splitting data...")
        self.imgs_train = self.imgs[:80]
        self.gt_imgs_train = self.gt_imgs[:80]
        self.imgs_validation = self.imgs[80:100]
        self.gt_imgs_validation = self.gt_imgs[80:100]
        print("Done!")
    
    def create_patches(self):
        """Create patches from the images."""
        print("Creating patches...")
        print("Creating patches for training set...")
        self.X_train, self.y_train = generate_samples(
            imgs=self.imgs_train,
            gt_imgs=self.gt_imgs_train,
            augm_patch_size=self.window_size,
            nb_samples=self.num_samples
        )
        print("Creating patches for validation set...")
        self.X_validation,self.y_validation = create_patches_test_set(
            images=self.imgs_validation,
            aug_patch_size=self.window_size,
            patch_size=self.patch_size,
            gt_imgs=self.gt_imgs_validation
        )
        print("Done!")

    def create_dataloader(self):
        """Create dataloader from the patches."""
        print("Creating dataloader...")
        self.train_dataloader = DataLoader(
                            dataset=list(zip(self.X_train, self.y_train)),
                            batch_size=self.batchsize,
                            shuffle=True,
                            num_workers=self.num_workers)

        self.validate_dataloader = DataLoader(
                                dataset=list(zip(self.X_validation, self.y_validation)),
                                batch_size=self.batchsize,
                                shuffle=False,
                                num_workers=self.num_workers)
        print("Done!")

    def proceed(self):
        """Proceed to the advanced processing."""
        self.load_data()
        if self.standardize:
            self.standardize_color()
        self.split_sets()
        self.create_patches()
        self.create_dataloader()
    