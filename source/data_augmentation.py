# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-27 -*-
# -*- python version : 3.11.6 -*-
# -*- Description: Create data augmentation-*-

# import libraries
import numpy as np
import random
import torch
from torchvision.transforms import v2
import constants
from scipy.ndimage import rotate

class DataAugmentation:
    """Class to create data augmentation."""
    def __init__(self, imgs, means, stds):
        self.mean_r, self.mean_g, self.mean_b = (*means,)
        self.std_r, self.std_g, self.std_b = (*stds,)
        self.imgs = imgs
    
    class AddGaussianNoise(object):
        """Add Gaussian noise to a tensor.
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def noisyImages(self):
        """Augmented images with these specifications:
        - To dtype float32
        - Normalize
        - Add Gaussian noise
        """
        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[self.mean_r, self.mean_g, self.mean_b], std=[self.std_r, self.std_g, self.std_b]),
            self.AddGaussianNoise(0., 0.5)
        ])
        transformed_imgs = []
        for img in self.imgs:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = transforms(img)
            img = img.permute(1,2,0)
            img = img.numpy()
            transformed_imgs.append(img)
        return np.array(transformed_imgs)

            
def create_augmented_patches(images, augm_patch_size):
    """Create augmented patches from images.
    Args:
        images (numpy array): Images to augment.
        augm_patch_size (int): Desired Size of the patch.
    Returns:
        numpy array: Augmented patches.
    """
    # Create augmented patches
    augmented_patches = []
    for im in images:
        two_dim = len(im.shape) < 3
        im_width = im.shape[0]
        im_height = im.shape[1]
        # Pad the image
        pad_size = (augm_patch_size - constants.PATCH_SIZE)//2
        padded = pading_img(im, pad_size)
        # Create patches
        for i in range(pad_size, im_height + pad_size, constants.PATCH_SIZE):
            for j in range(pad_size, im_width + pad_size, constants.PATCH_SIZE):
                if two_dim:
                    patch = padded[j-pad_size:j+constants.PATCH_SIZE+pad_size, i-pad_size:i+constants.PATCH_SIZE+pad_size]
                else:
                    patch = padded[j-pad_size:j+constants.PATCH_SIZE+pad_size, i-pad_size:i+constants.PATCH_SIZE+pad_size, :]
                augmented_patches.append(patch)
    augmented_patches = np.asarray(augmented_patches)
    return np.transpose(augmented_patches, (0, 3, 1, 2))

def pading_img(img, pad_size):
    """Pad an image.
    Args:
        img (numpy array): Image to pad.
        pad_size (int): Size of the padding.
    Returns:
        numpy array: Padded image.
    """
    # Pad the image
    if len(img.shape) < 3:
        padded = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)),  mode='reflect')
    else:
        padded = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    return padded

def create_augmented_patches_gt(images, gt_images, augm_patch_size):
    """Create augmented patches from images and ground truth images.
    Args:
        images (numpy array): Images to augment.
        gt_images (numpy array): Ground truth images to augment.
        augm_patch_size (int): Desired Size of the patch.
    Returns:
        numpy array: Augmented patches.
        numpy array: Augmented ground truth patches.
    """
    # Create augmented patches
    augmented_patches = []
    augmented_gt_patches = []
    for im, gt in zip(images, gt_images):
        two_dim = len(im.shape) < 3
        im_width = im.shape[0]
        im_height = im.shape[1]
        # Pad the image
        pad_size = (augm_patch_size - constants.PATCH_SIZE)//2
        padded_im = pading_img(im, pad_size)
        padded_gt = pading_img(gt, pad_size)
        # Create patches
        for i in range(pad_size, im_height + pad_size, constants.PATCH_SIZE):
            for j in range(pad_size, im_width + pad_size, constants.PATCH_SIZE):
                if two_dim:
                    im_patch = padded_im[j-pad_size:j+constants.PATCH_SIZE+pad_size, i-pad_size:i+constants.PATCH_SIZE+pad_size]   
                else:
                    im_patch = padded_im[j-pad_size:j+constants.PATCH_SIZE+pad_size, i-pad_size:i+constants.PATCH_SIZE+pad_size, :]
                label = 1 if (np.mean(padded_gt[j:j+constants.PATCH_SIZE, i:i+constants.PATCH_SIZE]) >  constants.FOREGROUND_THRESHOLD) else 0
                augmented_patches.append(im_patch)
                augmented_gt_patches.append(label)
    augmented_patches = np.asarray(augmented_patches)
    augmented_gt_patches = np.asarray(augmented_gt_patches)
    return np.transpose(augmented_patches, (0, 3, 1, 2)), augmented_gt_patches


def rotation(image, xy, angle):
    """Rotate an image.
    Args:
        image (numpy array): Image to rotate.
        xy (tuple): Coordinates of the rotation point.
        angle (int): Angle of the rotation.
    Returns:
        numpy array: Rotated image.
    """
    # Create rotation matrix using scipy function rotate
    image_rotate = rotate(image, angle, reshape=False)

    return image_rotate


def rotate_batch(img, gt, im_width, im_height, pad_size):
    # Rotates the entire batch for better performance
    random_rotation = 0
    if np.random.randint(0, 100) < 10:
        rotations = [90, 180, 270, 45, 135, 225, 315]
        random_rotation = np.random.randint(0, 7)
        img = rotation(img, np.array([im_width + 2*pad_size,im_height + 2* pad_size]), rotations[random_rotation])
        gt = rotation(gt, np.array([im_width, im_height]),rotations[random_rotation])
    return img, gt, random_rotation


def generate_single_sample(img, gt, pad_size, half_patch, augm_patch_size, random_rotation, im_width, im_height):
    x = np.empty((augm_patch_size, augm_patch_size, 3))
    y = np.empty((augm_patch_size, augm_patch_size, 3))

    if(random_rotation > 2):
        boundary = int((im_width- im_width / np.sqrt(2)) / 2)
    else : 
        boundary = 0
    center_x = np.random.randint(half_patch + boundary, im_width - half_patch - boundary)
    center_y = np.random.randint(half_patch + boundary, im_height - half_patch - boundary)

    x = img[center_x - half_patch:center_x + half_patch + 2 * pad_size,
           center_y - half_patch:center_y + half_patch + 2 * pad_size]
    y = gt[center_x - half_patch:center_x + half_patch,
           center_y - half_patch:center_y + half_patch]

    if np.random.randint(0, 2):  # vertical flip
        x = np.flipud(x)
    if np.random.randint(0, 2):  # horizontal flip
        x = np.fliplr(x)

    label = 1 if (np.array([np.mean(y)]) >  constants.FOREGROUND_THRESHOLD) else 0

    return x, label







def generate_batch(images, ground_truths, augm_patch_size, nb_batches, batch_size = 64, upsample = False):
    """Generate a batch of augmented patches.
    Args:
        images (numpy array): Images to augment.
        ground_truths (numpy array): Ground truth images to augment.
        augm_patch_size (int): Desired Size of the patch.
        nb_batches (int): Number of batches.
        batch_size (int): Size of the batch.
        upsample (bool): Upsample the batch.
    Returns:
        numpy array: Augmented patches.
        numpy array: Augmented ground truth patches.
    """
    np.random.seed(0)
    im_width = images[0].shape[0]
    im_height = images[0].shape[1]
    half_patch = constants.PATCH_SIZE // 2
    pad_size = (augm_patch_size - constants.PATCH_SIZE)//2
    padded_images = [pading_img(im, pad_size) for im in images]

    X = []
    Y = []

    for i in range(nb_batches):
        if i % 500 == 0:
            print(f'Create batch {i}/{nb_batches}')
        batch_in = []
        batch_out = []
        rand_ind = np.random.randint(0, len(images))

        img, gt, random_rotation = rotate_batch(padded_images[rand_ind], 
                                    ground_truths[rand_ind], im_width, im_height, pad_size)
        background_count = 0
        road_count = 0

        while len(batch_in) < batch_size :
            x, label = generate_single_sample(img, gt, pad_size, half_patch, augm_patch_size, random_rotation, im_width, im_height)
            if not upsample :
                batch_in.append(x)
                batch_out.append(label)
            elif label == 0:
                #For Background
                if background_count != batch_size // 2:
                    batch_in.append(x)
                    batch_out.append(label)
                    background_count += 1
            elif label == 1:
                #For road
                if road_count != batch_size // 2:
                    road_count += 1
                    batch_in.append(x)
                    batch_out.append(label)
        x_batch = np.array(batch_in)
        y_batch = np.array(batch_out)
        X.append(x_batch)
        Y.append(y_batch)
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(-1, augm_patch_size, augm_patch_size, 3)
    Y = Y.reshape(-1,)
    X = X.transpose(0, 3, 1, 2)
    return X, Y        

def generate_single_sample(img, gt, pad_size, half_patch, augm_batch_size, random_rotation, im_width, im_height):
    x = np.empty((augm_batch_size, augm_batch_size, 3))
    y = np.empty((augm_batch_size, augm_batch_size, 3))

    if(random_rotation > 2):
        boundary = int((im_width- im_width / np.sqrt(2)) / 2)
    else : 
        boundary = 0
    center_x = np.random.randint(half_patch + boundary, im_width - half_patch - boundary)
    center_y = np.random.randint(half_patch + boundary, im_height - half_patch - boundary)

    x = img[center_x - half_patch:center_x + half_patch + 2 * pad_size,
           center_y - half_patch:center_y + half_patch + 2 * pad_size]
    y = gt[center_x - half_patch:center_x + half_patch,
           center_y - half_patch:center_y + half_patch]

    if np.random.randint(0, 2):  # vertical flip
        x = np.flipud(x)
    if np.random.randint(0, 2):  # horizontal flip
        x = np.fliplr(x)

    label = 1 if (np.array([np.mean(y)]) >  constants.FOREGROUND_THRESHOLD) else 0

    return x, label
        






