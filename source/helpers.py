# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- Credits: Machine Learning course of the EPFL, Switzerland-*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Function to read and vizualize datas-*-

#import libraries
import numpy as np
import matplotlib.image as mpimg

# Helper functions
def load_image(infilename):
    """Load an image from disk.
    Args:
        infilename (str): Path to the image file.
    Returns:
        np.ndarray: The loaded image.
    """
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    """Convert float image to uint8.
    Args:
        img (np.ndarray): Image to convert.
    Returns:
        np.ndarray: Converted image.
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """Concatenate an image and its groundtruth.
    Args:
        img (np.ndarray): Image.
        gt_img (np.ndarray): Groundtruth image.
    Returns:
        np.ndarray: The concatenated image.
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    """Crop an image into patches of size w x h.
    Args:
        im (np.ndarray): Image to crop.
        w (int): Width of the patch.
        h (int): Height of the patch.
    Returns:
        list: List of patches.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches