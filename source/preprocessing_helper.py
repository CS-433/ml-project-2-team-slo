""" file containing all methods used in preprocessing data for the model """

import numpy as np
import os
import sys
import matplotlib.image as mpimg
import constants
import numpy
import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import rotate


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def create_windows(images, window_size):
    list_patches = []
    for im in images:
        is_2d = len(im.shape) < 3
        imgwidth = im.shape[0]
        imgheight = im.shape[1]
        padSize = (window_size - constants.PATCH_SIZE)//2
        padded = pad_image(im, padSize)
        for i in range(padSize, imgheight + padSize, constants.PATCH_SIZE):
            for j in range(padSize,imgwidth + padSize, constants.PATCH_SIZE):
                if is_2d:               
                    im_patch = padded[j-padSize:j+constants.PATCH_SIZE+padSize, i-padSize:i+constants.PATCH_SIZE+padSize]
                else:
                    im_patch = padded[j-padSize:j+constants.PATCH_SIZE+padSize, i-padSize:i+constants.PATCH_SIZE+padSize, :]
                list_patches.append(im_patch)
    list_patches = np.asarray(list_patches)
    return np.transpose(list_patches, (0, 3, 1, 2))

def create_windows_gt(images, gt_images, window_size):
    list_patches = []
    list_labels = []
    for im, gt in zip(images, gt_images):
        is_2d = len(im.shape) < 3
        imgwidth = im.shape[0]
        imgheight = im.shape[1]
        padSize = (window_size - constants.PATCH_SIZE)//2
        padded_im = pad_image(im, padSize)
        padded_gt = pad_image(gt, padSize)
        for i in range(padSize, imgheight + padSize, constants.PATCH_SIZE):
            for j in range(padSize,imgwidth + padSize, constants.PATCH_SIZE):
                if is_2d:               
                    im_patch = padded_im[j-padSize:j+constants.PATCH_SIZE+padSize, i-padSize:i+constants.PATCH_SIZE+padSize]   
                else:
                    im_patch = padded_im[j-padSize:j+constants.PATCH_SIZE+padSize, i-padSize:i+constants.PATCH_SIZE+padSize, :]
    
                label = 1 if (np.mean(padded_gt[j:j+constants.PATCH_SIZE, i:i+constants.PATCH_SIZE]) >  constants.FOREGROUND_THRESHOLD) else 0
                list_patches.append(im_patch)
                list_labels.append(label)
                
    list_patches = np.asarray(list_patches)
    return np.transpose(list_patches, (0, 3, 1, 2)), np.asarray(list_labels)

#pad an image 
def pad_image(img, padSize):
    is_2d = len(img.shape) < 3
    if is_2d:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize)),'reflect')
    else:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize),(0,0)),'reflect')
    
    
# source for rotation
#https://stackoverflow.com/questions/46657423/rotated-image-coordinates-after-scipy-ndimage-interpolation-rotate
def rot(image, xy, angle):
    im_rot = rotate(image,angle, reshape=False) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot

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

