# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-12-03 -*-
# -*- Last revision: 2023-12-03 -*-
# -*- python version : 3.11.6 -*-
# -*- Class dedicated to test data -*-

# import libraries
import numpy as np
from torch.utils.data import DataLoader

# import files
import constants
from load_datas import load_test_datas
from helpers import masks_to_submission, save_pred_as_png
from visualization import label_to_img
from data_augmentation import create_test_set


class TestData:
    """Class to process test datas"""

    def __init__(
        self,
        standardize=False,
        batchsize=constants.BATCH_SIZE,
        num_workers=constants.NUM_WORKERS,
        patch_size=constants.PATCH_SIZE,
        aug_patch_size=constants.AUG_PATCH_SIZE,
    ):
        """Constructor."""
        self.standardize = standardize
        self.batchsize = batchsize
        self.imgs = np.array([])
        self.test_dataloader = None
        self.num_workers = num_workers
        self.pred = None
        self.patch_size = patch_size
        self.aug_patch_size = aug_patch_size

    def load_data(self, test_path=constants.TEST_DIR):
        """Load the data."""
        print("Loading data...")
        self.imgs = load_test_datas(test_path)
        print("Done!")

    def create_dataloader(self):
        """Create dataloader from the patches."""
        print("Creating dataloader...")
        self.test_dataloader = DataLoader(
            dataset=self.patches, batch_size=self.batchsize, shuffle=False
        )
        print("Done!")

    def standardize_color(self):
        """Standardize the images."""
        print("Standardizing...")
        means = np.mean(self.imgs, axis=(0, 1, 2))
        stds = np.std(self.imgs, axis=(0, 1, 2))
        for i in range(constants.NUM_CHANNELS):
            self.imgs[:, :, :, i] = (self.imgs[:, :, :, i] - means[i]) / stds[i]
        print("Done!")

    def format_data(self):
        """Format the data to be used by the model."""
        print("Formatting data...")
        self.patches = create_test_set(self.imgs, self.aug_patch_size)
        print("Done!")

    def create_submission(self, pred):
        """
        Create the submission file.

        Args:
            pred (np.ndarray): The predictions.
        """
        images_filenames = save_pred_as_png(
            pred,
            len(self.imgs),
            self.patch_size,
            folder_path=constants.RESULTS_FOLDER_PATH,
        )

        masks_to_submission("submission.csv", *images_filenames)
        print("Submission created!")

    def proceed(self, test_path=constants.TEST_DIR):
        """Process all the steps."""
        self.load_data(test_path)
        if self.standardize:
            self.standardize_color()
        self.format_data()
        self.create_dataloader()
