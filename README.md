<div align="center">
<img src="./resources/logo-epfl.png" alt="Example Image" width="192" height="108">
</div>

<div align="center">
Ecole Polytechnique Fédérale de Lausanne
</div> 
<div align="center">
CS-433 Machine Learning
</div> 
<div align="center">
Road Segmentation
</div> 

# Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Data Wrangling](#Data-Wrangling)
- [Contributing](#contributing)
- [Models](#models)
- [Results](#results)

## Abstract 
The purpose of this project is to create a binary classifier that is able to recognize roads from Google maps satellite images. This kind of task is very common in image classification and computer vision. This repository presents solutions addressed to solve this problem. The major issues that have to be handled are the following:
*The datas are not well balanced : only 25% of the datas are roads
*Roads are majoritary vertical or horizontal
*The color of the roads is very similar to the one of sidewalk or parking area
   These problems will be discussed in the following sections.

## Project stucture
.
├── README.md
├── data
│   ├── images_report
│   ├── others
│   ├── sample_submission.csv
│   ├── submission
│   ├── test_set_images
│   │   ├── test_set_images.zip
│   └── training
│       ├── training.zip
│       └── untitled folder
├── source
│   ├── basic_model.py
│   ├── cnn.py
│   ├── constants.py
│   ├── data_augmentation.py
│   ├── data_processing.py
│   ├── helpers.py
│   ├── load_datas.py
│   ├── main.ipynb
│   ├── post_processing.py
│   ├── submission.csv
│   ├── training_utils.py
│   └── visualization.py
└── template
    ├── mask_to_submission.py
    ├── notes.md
    ├── segment_aerial_images.ipynb
    ├── submission_to_mask.py
    └── tf_aerial_images.py



The best solution can be run using the `run.py` file, in the source folder. The main results and the steps that leads to this solutions can be found in the main juypter notebook.

## Data wrangling
The datas consist of a set of 100 RGB images of size 400x400 pixels, comming with the correspond label images. The first step is to convert the images to arrays that can be used later on. The predictions are done on patches of size 16x16 pixels. The groundtruth images have to be croped to paches of this size. The corresponding images have to be croped the same way. 

## Data processing

To solve the problems mentionned in the abstract, the following solutions are proposed:</p>
* The color is normalized in order to obtain a homogeneous color around the training set.
* The datas are balanced to obtain an even distribution of road/background.

## Models

## Results

