# Notes on the code

## File segment_aerial_images.ipynb
 * Function `img_crop` : crop the input image into small patches of size `patch_size`. \
    It returns an array of size (nb_images, nb_patches, width_patch, height_patch, nb_channels)