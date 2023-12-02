# -*- coding: utf-8 -*-
# -*- author : fabio Palmisano - Yannis Laaroussi -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-12-01 -*-
# -*- python version : 3.11.6 -*-
# -*- Functions to post process datas -*-

# import libraries
from PIL import Image
import numpy as np

def save_pred_as_png(sub_preds, sub_imgs, patch_size, label_to_img):
    images_filenames = []

    nb_patches_per_img = len(sub_preds) // len(sub_imgs)

    for i, _ in enumerate(sub_imgs):
        # Assuming label_to_img returns a binary image (0s and 1s)

        predicted_im = label_to_img(608, 608, patch_size, patch_size, 
                                    sub_preds[i * nb_patches_per_img: (i + 1) * nb_patches_per_img])

        # Convert the numpy array to PIL Image
        predicted_img_pil = Image.fromarray((predicted_im * 255).astype(np.uint8))

        # Convert the image to RGB mode
        predicted_img_pil = predicted_img_pil.convert("RGB")

        # Save the image as PNG
        filename = f'output/preds_{i+1}.png'
        images_filenames.append(filename)
        predicted_img_pil.save(filename)
