# General notes
* The train images are of size 400x400 pixels
* The test images are of size 608x608 pixels
* All the images have 3 channels (R,G,B)
* The images are segmented into patches of size 16x16 (it gives 1444 patches for test images and 625 for train images)
# Notes on the code
## File segment_aerial_images.ipynb
* Function `img_crop` : crop the input image into small patches of size `patch_size`. \
    It returns an array of size (nb_images, nb_patches, width_patch, height_patch, nb_channels)
* Fuction `extract_img_features` compute the features of a given image (compute the mean color and the variance)
* 'foreground_threshold` define the threshold to determine if the patch is a road or not
* Function `label_to_img` transform the labels into an image
* Function `make_img_overlay` puts the predicted mask on top of the original image
