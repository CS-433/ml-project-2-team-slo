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

            
def create_augmented_patches(images, patch_size):
    """Create augmented patches from images.
    Args:
        images (numpy array): Images to augment.
        patch_size (int): Desired Size of the patch.
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
        pad_size = (patch_size - constants.PATCH_SIZE)//2
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

def create_augmented_patches_gt(images, gt_images, patch_size):
    """Create augmented patches from images and ground truth images.
    Args:
        images (numpy array): Images to augment.
        gt_images (numpy array): Ground truth images to augment.
        patch_size (int): Desired Size of the patch.
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
        pad_size = (patch_size - constants.PATCH_SIZE)//2
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
    # Create rotation center
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(image_rotate.shape[:2][::-1])-1)/2.
    # Create rotation matrix
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Create new coordinates
    org = xy-org_center
    new = np.array([org[0]*np.cos(angle) + org[1]*np.sin(angle), -org[0]*np.sin(angle) + org[1]*np.cos(angle)])
    # Create new image
    new_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x, y = i - xy[0], j - xy[1]
            x, y = np.dot(rot_mat, np.array([x, y]))
            x, y = int(x + xy[0]), int(y + xy[1])
            if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
                new_image[i, j] = image_rotate[x, y]
    return new_image
