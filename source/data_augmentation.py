# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-27 -*-
# -*- python version : 3.11.6 -*-
# -*- Description: Create data augmentation-*-

# import libraries
import numpy as np
import random
import torch
from torchvision.transforms import v2

class DataAugmentation:
    """Class to create data augmentation."""
    def __init__(self, imgs, means, stds):
        self.mean_r, self.mean_g, self.mean_b = (*means,)
        self.std_r, self.std_g, self.std_b = (*stds,)
        self.imgs = imgs
    
    class AddGaussianNoise(object):
        """Add Gaussian noise to a tensor.
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def noisyImages(self):
        """Augmented images with these specifications:
        - To dtype float32
        - Normalize
        - Add Gaussian noise
        """
        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[self.mean_r, self.mean_g, self.mean_b], std=[self.std_r, self.std_g, self.std_b]),
            self.AddGaussianNoise(0., 0.5)
        ])
        transformed_imgs = []
        for img in self.imgs:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = transforms(img)
            img = img.permute(1,2,0)
            img = img.numpy()
            transformed_imgs.append(img)
        return np.array(transformed_imgs)

            
