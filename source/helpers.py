# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit, Yannis Laaroussi -*-
# -*- Credits: Machine Learning course of the EPFL, Switzerland-*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Function to read and vizualize datas-*-

# import libraries
import numpy as np
import matplotlib.image as mpimg
import constants
import re
import os
from PIL import Image
from visualization import label_to_img


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


def value_to_class(patch):
    patch_mean = np.mean(patch)
    if patch_mean > constants.FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


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


def f1_score(preds, targets):
    """
    Compute the F1 score.

    Args:
        preds (np.ndarray): The predictions
        targets (np.ndarray): The targets

    Returns:
        float: The F1 score
    """
    assert len(preds) == len(targets)

    tp = np.sum(preds * targets)
    fp = np.sum(preds * (1 - targets))
    fn = np.sum((1 - preds) * targets)
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    if precision + recall == 0:
        return np.nan
    return 2 * (precision * recall) / (precision + recall)


def save_pred_as_png(
    sub_preds,
    nb_imgs,
    patch_size=constants.PATCH_SIZE,
    folder_path=constants.RESULTS_FOLDER_PATH,
):
    """
    Save the predictions as PNG

    Args:
        sub_preds (np.ndarray): The predictions
        nb_imgs (int): The number of images
        patch_size (int): The size of the patch
        folder_path (str): The path to the folder where to save the images

    Returns:
        list: The list of the images filenames
    """
    images_filenames = []

    nb_patches_per_img = len(sub_preds) // nb_imgs

    if not os.path.exists(folder_path):
        # Create the folder if it does not exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        # Do nothing if the folder already exists
        print(f"Folder '{folder_path}' already exists.")

    for i in range(nb_imgs):
        # Assuming label_to_img returns a binary image (0s and 1s)

        predicted_im = label_to_img(
            608,
            608,
            patch_size,
            patch_size,
            sub_preds[i * nb_patches_per_img : (i + 1) * nb_patches_per_img],
        )

        # Convert the numpy array to PIL Image
        predicted_img_pil = Image.fromarray((predicted_im * 255).astype(np.uint8))

        # Convert the image to RGB mode
        predicted_img_pil = predicted_img_pil.convert("RGB")

        # Save the image as PNG
        filename = os.path.join(folder_path, f"preds_{i+1}.png")
        images_filenames.append(filename)
        predicted_img_pil.save(filename)

    return images_filenames


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", os.path.basename(image_filename)).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i : i + patch_size, j : j + patch_size]
            label = value_to_class(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for fn in image_filenames[0:]:
            f.writelines("{}\n".format(s) for s in mask_to_submission_strings(fn))
