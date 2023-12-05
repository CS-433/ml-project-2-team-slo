# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-27 -*-
# -*- python version : 3.11.6 -*-
# -*- Description: Create data augmentation-*-

# import libraries
import numpy as np
from torchvision.transforms import v2
import constants
from scipy.ndimage import rotate
from helpers import value_to_class, img_crop
            
def create_patches_test_set(images,aug_patch_size,patch_size,gt_imgs = None):
    """Create augmented patches from images.
    Args:
        images (numpy array): Images to augment.
        augm_patch_size (int): Desired Size of the patch.
    Returns:
        numpy array: Augmented patches.
    """
    imgwidth = images.shape[1]
    imgheight = images.shape[2]
    pad_size = (aug_patch_size - patch_size)//2
    aug_images =np.array([pading_img(im,pad_size) for im in images])
    list_patches = []
    for img in aug_images:
        for i in range(pad_size, imgheight + pad_size, patch_size):
            for j in range(pad_size,imgwidth + pad_size, patch_size):
                im_patch = img[j-pad_size:j+patch_size+pad_size, i-pad_size:i+patch_size+pad_size]
                list_patches.append(im_patch)
    patches = np.asarray(list_patches)
    patches = np.transpose(patches, (0, 3, 1, 2))
    if gt_imgs is not None:
        gt_patches = [img_crop(gt_imgs[i],patch_size,patch_size) for i in range(len(gt_imgs))]
        gt_patches = np.asarray(
                [
                    gt_patches[i][j]
                    for i in range(len(gt_patches))
                    for j in range(len(gt_patches[i]))
                ]
            )
        labels = np.asarray([value_to_class(np.mean(patch)) for patch in gt_patches])
    return patches,labels

def pading_img(img, pad_value,gt=False):
    """Pad an image.
    Args:
        img (numpy array): Image to pad.
        pad_size (int): Size of the padding.
    Returns:
        numpy array: Padded image.
    """
    # Pad the image
    if gt:
        img_pad = np.pad(img, ((pad_value, pad_value), (pad_value, pad_value)),  mode='reflect')
    else:
        img_pad = np.pad(img, ((pad_value, pad_value), (pad_value, pad_value), (0, 0)), mode='reflect')
        
    return img_pad


def generate_samples(imgs, gt_imgs, augm_patch_size, nb_samples):
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
    half_patch = constants.PATCH_SIZE // 2
    pad_size = (augm_patch_size - constants.PATCH_SIZE)//2
    padded_images = [pading_img(im, pad_size) for im in imgs]

    features = []
    labels = []

    nb_road = 0
    nb_background = 0
    count = 0
    img_index = np.random.randint(0, len(imgs))
    while len(labels) < nb_samples:
        if(count % 32 == 0):
            img_index = np.random.randint(0, len(imgs))
        img,gt = padded_images[img_index], gt_imgs[img_index]
        rotate_prob = np.random.rand()
        if rotate_prob <= 0.1:
            angle = np.random.randint(20, 45)
            img = rotate(img, angle=angle, reshape=False)
            boundary = int((img.shape[0] - img.shape[1] / np.sqrt(2)) / 2)
            img = img[boundary:-boundary, boundary:-boundary,:]
            gt = rotate(gt, angle=angle, reshape=False)
            gt = gt[boundary:-boundary, boundary:-boundary]
        feature, label = generate_single_sample(img,gt, pad_size,half_patch)
        if label == 1:
            if nb_road < nb_samples//2:
                features.append(feature)
                labels.append(label)
                nb_road += 1
        else:
            if nb_background < nb_samples//2:
                features.append(feature)
                labels.append(label)
                nb_background += 1
        count += 1
    features = np.transpose(features, (0, 3, 1, 2))
    return np.asarray(features), np.asarray(labels)      

def generate_single_sample(img, gt, pad_size, half_patch):
    im_width = gt.shape[0]
    im_height = gt.shape[1]
    x_coord = np.random.randint(half_patch+pad_size, im_width - half_patch - pad_size)
    y_coord = np.random.randint(half_patch+pad_size, im_height - half_patch - pad_size)
    patch_coord = {'x': x_coord, 'y': y_coord}

    features = img[patch_coord['x'] - half_patch-pad_size:patch_coord['x'] + half_patch + pad_size,
           patch_coord['y'] - half_patch-pad_size:patch_coord['y'] + half_patch + pad_size]
    y = gt[patch_coord['x'] - half_patch:patch_coord['x'] + half_patch,
           patch_coord['y'] - half_patch:patch_coord['y'] + half_patch]
    label = value_to_class(y)
    return features, label