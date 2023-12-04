# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-12-03 -*-
# -*- Last revision: 2023-12-03 -*-
# -*- python version : 3.11.6 -*-
# -*- Class dedicated to test data -*-

#import libraries
import numpy as np
from torch.utils.data import DataLoader

#import files
import constants
from load_datas import load_test_datas
from helpers import masks_to_submission
from post_processing import save_pred_as_png
from visualization import label_to_img


class TestData:
    """ Class to process test datas"""
    def __init__(
        self,
        model,
        batchsize=constants.BATCH_SIZE,
        num_workers=constants.NUM_WORKERS,
        patch_size = constants.PATCH_SIZE
    ):
        """Constructor."""
        self.batchsize = batchsize
        self.imgs = np.array([])
        self.test_dataloader = None
        self.num_workers = num_workers
        self.pred = None
        self.patch_size = patch_size

    def load_data(self):
        """Load the data."""
        print("Loading data...")
        self.imgs = load_test_datas()
        print("Done!")
    
    def create_dataloader(self):
        """Create dataloader from the patches."""
        print("Creating dataloader...")
        self.test_dataloader = DataLoader(
                                dataset=self.imgs,
                                batch_size=self.batchsize,
                                shuffle=False)
        print("Done!")
    
    def prediction(self,model):
        """Create prediction from the model."""
        print("Creating prediction...")
        self.pred = model.predict(self.test_dataloader)
        print("Done!")

    def create_submission(self):
        images_filenames = save_pred_as_png(self.pred, len(self.imgs), self.patch_size, label_to_img)
        masks_to_submission('submission.csv', *images_filenames)
    
    def proceed(self):
        """Process all the steps."""
        self.load_data()
        self.create_dataloader()
        self.prediction()
        self.create_submission()
        print("Submission created!")