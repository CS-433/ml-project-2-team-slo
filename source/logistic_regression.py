# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-12-03 -*-
# -*- Credits: Machine Learning course of the EPFL, Switzerland -*-
# -*- Last revision: 2023-12-13 (Vincent Roduit, Yannis Laaroussi) -*-
# -*- python version : 3.11.6 -*-
# -*- Description: Logistic regression class-*-

# import libraries
import numpy as np
from sklearn import linear_model
from helpers import f1_score, value_to_class


class LogisticRegression:
    """Class for model Logistic Regression."""

    def __init__(self, imgs, gt_imgs):
        """Constructor."""
        self.imgs_patches = imgs
        self.gt_imgs_patches = gt_imgs
        self.X = np.array([])
        self.Y = np.array([])

    def extract_features(self, img):
        """Extract features from an image.
        Args:
            img (np.array): Image to extract features from.
        Returns:
            np.array: Array of features.
        """
        feat_m = np.mean(img, axis=(0, 1))
        feat_v = np.var(img, axis=(0, 1))
        feat = np.append(feat_m, feat_v)
        return feat

    def extract_features_2d(self, img):
        """Extract 2D features from an image.
        Args:
            img (np.array): Image to extract features from.
        Returns:
            np.array: Array of features.
        """
        feat_m = np.mean(img)
        feat_v = np.var(img)
        feat = np.append(feat_m, feat_v)
        return feat

    def compute_vectors(self):
        """Compute the vectors X and Y. """
        self.X = np.asarray(
            [
                self.extract_features_2d(self.imgs_patches[i])
                for i in range(len(self.imgs_patches))
            ]
        )
        self.Y = np.asarray(
            [
                value_to_class(np.mean(self.gt_imgs_patches[i]))
                for i in range(len(self.gt_imgs_patches))
            ]
        )

    def train(self):
        """Train the model."""
        self.logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
        self.logreg.fit(self.X, self.Y)

    def predict(self):
        """Predict the labels."""
        self.prediction = self.logreg.predict(self.X)
        self.accuracy = self.logreg.score(self.X, self.Y)
        self.f1 = f1_score(self.Y, self.prediction)
