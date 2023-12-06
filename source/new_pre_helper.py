import numpy as np
import os
import sys
import matplotlib.image as mpimg
import constants
import numpy
import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import rotate
from visualization import visualize_patch
from helpers import *

def new_create_windows_gt(images, gt_images, augm_patch_size):
    # Fonction testée et donne les mêmes résultats que create_windows_gt
    list_patches = []
    list_labels = []
    for im in images:
        imgwidth = im.shape[0]
        imgheight = im.shape[1]
        padSize = (augm_patch_size - constants.PATCH_SIZE)//2
        padded_im = pad_image(im, padSize)
        for i in range(padSize, imgheight + padSize, constants.PATCH_SIZE):
            for j in range(padSize,imgwidth + padSize, constants.PATCH_SIZE): 
                im_patch = padded_im[j-padSize:j+constants.PATCH_SIZE+padSize, i-padSize:i+constants.PATCH_SIZE+padSize, :]
                list_patches.append(im_patch)
    gt_patches = [img_crop(gt, constants.PATCH_SIZE, constants.PATCH_SIZE) for gt in gt_images]
    gt_patches = [patch for patches in gt_patches for patch in patches]
    list_labels = [value_to_class(np.mean(patch)) for patch in gt_patches]
    list_labels = np.asarray(list_labels)
    list_patches = np.asarray(list_patches)
    return np.transpose(list_patches, (0, 3, 1, 2)), list_labels

def pad_image(img, padSize):
    return np.pad(img,((padSize,padSize),(padSize,padSize),(0,0)),'reflect')


def new_image_generator(imgs, gt_imgs,rotated_imgs,rotated_gt_imgs, window_size,num_samples, batch_size):
    np.random.seed(0)
    imgWidth = imgs[0].shape[0]
    imgHeight = imgs[0].shape[1]
    half_patch = constants.PATCH_SIZE // 2
    
    padSize = (window_size - constants.PATCH_SIZE) // 2
    aug_imgs = [pad_image(image,padSize) for image in imgs]

    X = []
    Y = []
    nb_batches = num_samples // batch_size
    for i in range(nb_batches):
        if i % (nb_batches/10) == 0:
            print(f'Batch {i+1}/{nb_batches}')
        list_patches = []
        list_labels = [] 
        boundary = 0
        img, gt,boundary = choose_image(rotated_imgs,rotated_gt_imgs,aug_imgs,gt_imgs)
        gt_count = 0
        img_count = 0
        count = 0
        while len(list_patches) < batch_size:
            if count == 10 * batch_size:
                img, gt,boundary = choose_image(rotated_imgs,rotated_gt_imgs,aug_imgs,gt_imgs)
                count = 0
            img_patch, label = create_sample(img,gt,half_patch,boundary,padSize)
            if label == 0:
                if gt_count != batch_size // 2:
                    list_patches.append(img_patch)
                    list_labels.append(label)
                    gt_count += 1
            elif label == 1:
                if img_count != batch_size // 2:
                    img_count += 1
                    list_patches.append(img_patch)
                    list_labels.append(label)
            count += 1
                
        list_patches = np.array(list_patches)
        list_labels = np.array(list_labels )

        X.append(list_patches)
        Y.append(list_labels)
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(-1, window_size, window_size, 3)
    Y = Y.reshape(-1,)
    X = X.transpose(0, 3, 1, 2)
    print('end process...')
    return X, Y

def rotate_imgs_train(imgs_train, gt_imgs_train):
    angles = [45,135,225,315]
    rotated_imgs = []
    rotated_gt_imgs = []
    for angle in angles:
        print(f'Rotation for {angle} degrees')
        for img,gt in zip(imgs_train,gt_imgs_train):
            rotated_imgs.append(rotate(img, angle))
            rotated_gt_imgs.append(rotate(gt, angle))
    rotated_imgs = np.asarray(rotated_imgs)
    rotated_gt_imgs = np.asarray(rotated_gt_imgs)
    return rotated_imgs, rotated_gt_imgs

def choose_image(rotated_imgs,rotated_gt_imgs,aug_imgs,gt_imgs):
    if (np.random.randn() <= 0.1):
        img_index = np.random.randint(len(rotated_imgs))
        img = rotated_imgs[img_index]
        gt = rotated_gt_imgs[img_index]
        boundary = int((img.shape[0] - img.shape[0] / np.sqrt(2)) / 2)
    else:
        img_index = np.random.randint(len(aug_imgs))
        img = aug_imgs[img_index]
        gt = gt_imgs[img_index]
        boundary = 0
    return img, gt,boundary

def create_sample(img,gt,half_patch,center_limit,size_pading):
    img_width = gt.shape[0]
    img_height = gt.shape[1]
    x_coord = np.random.randint(half_patch + center_limit, img_width - half_patch - center_limit)
    y_coord = np.random.randint(half_patch + center_limit, img_height - half_patch - center_limit)
    patch_coord = {'x':x_coord,'y':y_coord}
    aug_patch_coord = {'x':x_coord + size_pading,'y':y_coord + size_pading}
    img_patch = img[aug_patch_coord['x'] - half_patch-size_pading:aug_patch_coord['x'] + half_patch + size_pading,
            aug_patch_coord['y']  - half_patch - size_pading: aug_patch_coord['y'] + half_patch + size_pading]
    gt_patch = gt[patch_coord['x'] - half_patch : patch_coord['x'] + half_patch,
            patch_coord['y'] - half_patch : patch_coord['y'] + half_patch]
    label = value_to_class(np.mean(gt_patch))
    return img_patch, label