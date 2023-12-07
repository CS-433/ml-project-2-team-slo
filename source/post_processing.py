# -*- coding: utf-8 -*-
# -*- author : Fabio Palmisano - Yannis Laaroussi -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-12-01 -*-
# -*- python version : 3.11.6 -*-
# -*- Functions to post process datas -*-

# import libraries
from PIL import Image
import numpy as np
import os

def save_pred_as_png(sub_preds,nb_imgs,patch_size,label_to_img,folder_path):
    images_filenames = []

    nb_patches_per_img = len(sub_preds) // nb_imgs
    
    import os
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
      print(f"Folder '{folder_path}' created successfully.")
    else:
      print(f"Folder '{folder_path}' already exists.")

    for i in range(nb_imgs):
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
    return images_filenames