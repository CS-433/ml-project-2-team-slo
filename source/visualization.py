# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Functions to visualize datas -*-

#import librairies
import matplotlib.pyplot as plt
from PIL import Image

#import files
from constants import*
from helpers import*

def visualize(imgs, gt_imgs, index=0):
    """Visualize an image and its groundtruth.
    Args:
        imgs (np.ndarray): Images.
        gt_imgs (np.ndarray): Groundtruth images.
    """
    cimg = concatenate_images(imgs[index], gt_imgs[index])
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg, cmap="Greys_r")

def visualize_patch(img):
    """ Visualize a patch.
    Args:
        img: Image.
    """
    fig1 = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="Greys_r")

def visualize_solution(img, idx_img, prediction):
    """Visualize an image and the predicted groundtruth.
    Args:
        img: Image.
        imgwidth: Image width.
        imgheight: Image height.
        prediction: Prediction.

    """
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    predicted_im = label_to_img(imgwidth, imgheight, PATCH_SIZE, PATCH_SIZE, prediction, idx_img = idx_img)
    fig1 = plt.figure(figsize=(7, 7))  # create a figure with the default size

    image = make_img_overlay(img, predicted_im)

    plt.imshow(image)


def label_to_img(imgwidth, imgheight, w, h, labels, idx_img = 1):
    """Convert labels to images.
    Args:
        imgwidth (int): Image width.
        imgheight (int): Image height.
        w (int): Width.
        h (int): Height.
        labels (np.ndarray): Labels.
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 1444 * idx_img 
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im


def make_img_overlay(img, predicted_img):
    """Make an image overlay.
    Args:
        img (np.ndarray): Image.
        predicted_img (np.ndarray): Predicted image.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)

    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
    