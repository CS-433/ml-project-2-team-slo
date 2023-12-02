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
from scipy.ndimage import rotate

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
        index_0 = np.where(self.gt_imgs_train == 0)[0]
        index_1 = np.where(self.gt_imgs_train == 1)[0]
        # Get the number of patches
        n_0 = len(index_0)
        n_1 = len(index_1)
        # Get the difference between the number of patches
        diff = n_0 - n_1
        # Get the indexes of the patches to remove
        index_to_remove = np.random.choice(index_0, diff, replace=False)
        # Remove the patches
        self.imgs_train = np.delete(self.imgs_train, index_to_remove, axis=0)
        self.gt_imgs_train = np.delete(self.gt_imgs_train, index_to_remove)
        print("Done!")
    
    def image_generator(images, ground_truths, window_size,nb_batches, batch_size = 64,upsample=False):
        np.random.seed(0)
        imgWidth = images[0].shape[0]
        imgHeight = images[0].shape[1]
        half_patch = constants.PATCH_SIZE // 2
        
        padSize = (window_size - constants.PATCH_SIZE) // 2
        paddedImages = []
        for image in images:
            paddedImages.append(pad_image(image,padSize))
        X = []
        Y = []
        for _ in range(nb_batches):
            batch_input = []
            batch_output = [] 
            
            #rotates the whole batch for better performance
            randomIndex = np.random.randint(0, len(images))
            print(randomIndex)  
            img = paddedImages[randomIndex]
            gt = ground_truths[randomIndex]
            # rotate with probability 10 / 100
            random_rotation = 0
            if (np.random.randint(0, 100) < 10):
                rotations = [90, 180, 270, 45, 135, 225, 315]
                random_rotation = np.random.randint(0, 7)
                img = rot(img, np.array([imgWidth+2*padSize, imgHeight+2*padSize]), rotations[random_rotation])
                gt = rot(gt, np.array([imgWidth, imgHeight]), rotations[random_rotation]) 
            
            background_count = 0
            road_count = 0
            while len(batch_input) < batch_size:
                x = np.empty((window_size, window_size, 3))
                y = np.empty((window_size, window_size, 3))
                
                
                # we need to limit possible centers to avoid having a window in an interpolated part of the image
                # we limit ourselves to a square of width 1/sqrt(2) smaller
                if(random_rotation > 2):
                    boundary = int((imgWidth - imgWidth / np.sqrt(2)) / 2)
                else:
                    boundary = 0
                center_x = np.random.randint(half_patch + boundary, imgWidth  - half_patch - boundary)
                center_y = np.random.randint(half_patch + boundary, imgHeight - half_patch - boundary)
                
                x = img[center_x - half_patch:center_x + half_patch + 2 * padSize,
                        center_y  - half_patch:center_y + half_patch + 2 * padSize]
                y = gt[center_x - half_patch : center_x + half_patch,
                    center_y - half_patch : center_y + half_patch]
                
                # vertical
                if(np.random.randint(0, 2)):
                    x = np.flipud(x)
                
                # horizontal
                if(np.random.randint(0, 2)):
                    x = np.fliplr(x)
                label = 1 if (np.array([np.mean(y)]) >  constants.FOREGROUND_THRESHOLD) else 0
                
                # makes sure we have an even distribution of road and non road if we oversample
                if not upsample:
                    batch_input.append(x)
                    batch_output.append(label)
                elif label == 0:
                    # case background
                    if background_count != batch_size // 2:
                        batch_input.append(x)
                        batch_output.append(label)
                        background_count += 1
                elif label == 1:
                    # case road
                    if road_count != batch_size // 2:
                        road_count += 1
                        batch_input.append(x)
                        batch_output.append(label)
                    
            batch_x = np.array( batch_input )
            batch_y = np.array( batch_output )

            X.append(batch_x)
            Y.append(batch_y)
        X = np.array(X)
        Y = np.array(Y)
        X = X.reshape(-1, window_size, window_size, 3)
        Y = Y.reshape(-1,)
        X = X.transpose(0, 3, 1, 2)
        return X, Y