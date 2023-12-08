import numpy as np
import constants
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from helpers import *

def create_test_set(images, aug_patch_size, gt_images=None):
    # Fonction testée et donne les mêmes résultats que create_windows_gt
    list_patches = []
    list_labels = []
    for im in images:
        imgwidth = im.shape[0]
        imgheight = im.shape[1]
        size_padding = (aug_patch_size - constants.PATCH_SIZE)//2
        padded_im = pad_image(im, size_padding)
        for i in range(size_padding, imgheight + size_padding, constants.PATCH_SIZE):
            for j in range(size_padding,imgwidth + size_padding, constants.PATCH_SIZE): 
                im_patch = padded_im[j-size_padding:j+constants.PATCH_SIZE+size_padding, i-size_padding:i+constants.PATCH_SIZE+size_padding, :]
                list_patches.append(im_patch)
    list_patches = np.asarray(list_patches)
    if gt_images is not None:
        gt_patches = [img_crop(gt_img, constants.PATCH_SIZE, constants.PATCH_SIZE) for gt_img in gt_images]
        gt_patches = [patch for patches in gt_patches for patch in patches]
        list_labels = [value_to_class(patch) for patch in gt_patches]
        list_labels = np.asarray(list_labels)
    else:
        return np.transpose(list_patches, (0, 3, 1, 2))
    return np.transpose(list_patches, (0, 3, 1, 2)), list_labels

def pad_image(img, padSize):
    return np.pad(img,((padSize,padSize),(padSize,padSize),(0,0)),'reflect')


def create_samples(imgs, gt_imgs, aug_patch_size,num_samples, batch_size):
    np.random.seed(0)
    half_patch = constants.PATCH_SIZE // 2
    
    size_padding = (aug_patch_size - constants.PATCH_SIZE) // 2
    aug_imgs = [pad_image(image,size_padding) for image in imgs]

    X = []
    Y = []
    rotated_imgs, rotated_gt_imgs = rotate_imgs_train(aug_imgs, gt_imgs)
    blured_imgs = blur_images(aug_imgs)
    nb_batches = num_samples // batch_size
    for i in range(nb_batches):
        if i % (nb_batches/10) == 0:
            print(f'Batch {i+1}/{nb_batches}')
        list_patches = []
        list_labels = [] 

        img, gt = choose_image(rotated_imgs,rotated_gt_imgs,aug_imgs,gt_imgs,blured_imgs)

        gt_count = 0 #background
        img_count = 0 #road
        count = 0

        while len(list_patches) < batch_size:
            if count == 10 * batch_size:
                img, gt = choose_image(rotated_imgs,rotated_gt_imgs,aug_imgs,gt_imgs,blured_imgs)
                count = 0

            img_patch, label = create_single_sample(img,gt,half_patch,size_padding)

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
    X = X.reshape(-1, aug_patch_size, aug_patch_size, 3)
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
            rotated_imgs.append(rotate(img, angle, reshape=False, mode='mirror'))
            rotated_gt_imgs.append(rotate(gt, angle, reshape=False, mode='mirror'))
    rotated_imgs = np.asarray(rotated_imgs)
    rotated_gt_imgs = np.asarray(rotated_gt_imgs)
    return rotated_imgs, rotated_gt_imgs

def blur_images(imgs_train,sigma=constants.SIGMA):
    blurred_imgs = []

    for img in imgs_train:
        blurred_image = np.zeros_like(img)
        for channel in range(img.shape[2]):
            blurred_image[:, :, channel] = gaussian_filter(img[:, :, channel], sigma)
        blurred_imgs.append(blurred_image)
    blurred_imgs = np.asarray(blurred_imgs)

    return blurred_imgs

def choose_image(rotated_imgs,rotated_gt_imgs,aug_imgs,gt_imgs,blured_imgs):
    if (np.random.randn() <= 0.1):
        img_index = np.random.randint(len(rotated_imgs))
        img = rotated_imgs[img_index]
        gt = rotated_gt_imgs[img_index]
    elif (np.random.randn() > 0.9):
        img_index = np.random.randint(len(blured_imgs))
        img = blured_imgs[img_index]
        gt = gt_imgs[img_index]
    else:
        img_index = np.random.randint(len(aug_imgs))
        img = aug_imgs[img_index]
        gt = gt_imgs[img_index]
    return img, gt

def create_single_sample(img,gt,half_patch,size_pading):
    img_width = gt.shape[0]
    img_height = gt.shape[1]
    x_coord = np.random.randint(half_patch, img_width  - half_patch)
    y_coord = np.random.randint(half_patch, img_height - half_patch)
    patch_coord = {'x': x_coord, 'y': y_coord}
    img = img[patch_coord['x'] - half_patch:patch_coord['x'] + half_patch + 2 * size_pading,
            patch_coord['y']  - half_patch:patch_coord['y'] + half_patch + 2 * size_pading]
    gt_img = gt[patch_coord['x'] - half_patch : patch_coord['x'] + half_patch,
            patch_coord['y'] - half_patch : patch_coord['y'] + half_patch]
    label = value_to_class(np.mean(gt_img))
    return img, label