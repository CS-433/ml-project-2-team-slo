# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-35 -*-
# -*- python version : 3.11.6 -*-
# -*- Functions to visualize datas -*-

#import librairies
import matplotlib.pyplot as plt

#import files
from constants import*
from helpers import*

def visualize(imgs, gt_imgs, index=0):
    """Visualize images and groundtruths.
    Args:
        imgs (np.ndarray): Images.
        gt_imgs (np.ndarray): Groundtruth images.
    """
    cimg = concatenate_images(imgs[0], gt_imgs[0])
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg, cmap="Greys_r")