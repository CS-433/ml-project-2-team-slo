<div align="center">
<img src="./resources/logo-epfl.png" alt="Example Image" width="192" height="108">
</div>

<div align="center">
Ecole Polytechnique Fédérale de Lausanne
</div> 
<div align="center">
CS-433 Machine Learning
</div> 

# Road segmentation from satellite images

## Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Data Wrangling](#data-wrangling)
- [Models](#models)
- [Results](#results)
- [Run the solution](#run-the-solution)

## Abstract 
The purpose of this project is to create a binary classifier that is able to recognize roads from Google maps satellite images. This kind of task is very common in image classification and computer vision. This repository presents solutions addressed to solve this problem. The major issues that have to be handled are the following:
*The datas are not well balanced : only 25% of the datas are roads
*Roads are majoritary vertical or horizontal
*The color of the roads is very similar to the one of sidewalk or parking area
These problems will be discussed in the following sections.

## Project structure
```
.
├── README.md
├── models
│   ├── advanced_cnn_128.pth
│   ├── advanced_cnn_128_adamw.pth
│   ├── advanced_cnn_128_blur.pth
│   ├── advanced_cnn_128_blur_025.pth
│   ├── advanced_cnn_128_nesterov.pth
│   ├── advanced_cnn_128_thr_02.pth
│   ├── advanced_cnn_128_thr_03.pth
│   ├── advanced_cnn_64.pth
│   ├── advanced_cnn_color_128.pth
│   ├── basic_cnn_16.pth
│   ├── basic_cnn_32.pth
│   └── basic_cnn_64.pth
├── resources
│   └── logo-epfl.png
├── source
│   ├── __pycache__
│   ├── cnn.py
│   ├── constants.py
│   ├── data_augmentation.py
│   ├── data_processing.py
│   ├── helpers.py
│   ├── load_datas.py
│   ├── logistic_regression.py
│   ├── main.ipynb
│   ├── post_processing.py
│   ├── run.py
│   ├── test_data.py
│   └── visualization.py
└── template
    ├── mask_to_submission.py
    ├── notes.md
    ├── segment_aerial_images.ipynb
    ├── submission_to_mask.py
    └── tf_aerial_images.py
```


The best solution can be run using the `run.py` file, in the source folder. The main results and the steps that leads to this solutions can be found in the main juypter notebook.

## Data wrangling
The datas consist of a set of 100 RGB images of size 400x400 pixels, comming with the correspond label images. The first step is to convert the images to arrays that can be used later on. The predictions are done on patches of size 16x16 pixels. The groundtruth images have to be croped to paches of this size. The corresponding images have to be croped the same way. 

## Data processing

To solve the problems mentionned in the abstract, the following solutions are proposed:</p>
* The color is normalized in order to obtain homogeneous RGB images for the training set.
* The datas are balanced to obtain an even distribution of road/background.
* The size of the patch feeded to the neural network can be changed to obtain better performance.

## Models
Two models are proposed for this project, a basic and an advanced convolution neural network.
## Results

| Model                              | Patch size | Optimizer    | Threshold | Accuracy | F1-score | AICrowd F1-Score | AICrowd accuracy |
|------------------------------------|------------|--------------|-----------|----------|----------|------------------|------------------|
| Basic                              | 16         | Adam         | 0.25      | 0.800    | 0.679    | -                | -                |
| Basic                              | 32         | Adam         | 0.25      | 0.832    | 0.720    | -                | -                |
| Basic                              | 64         | Adam         | 0.25      | 0.850    | 0.743    | -                | -                |
| Advanced                           | 64         | Adam         | 0.25      | 0.889    | 0.792    | -                | -                |
| Advanced                           | 128        | Adam         | 0.25      | 0.912    | 0.833    | 0.856            | 0.921            |
| Advanced (+ color standardization) | 128        | Adam         | 0.25      | 0.924    | 0.857    | 0.868            | 0.929            |
| Advanced (+ color standardization) | 128        | AdamW        | 0.25      | 0.921    | 0.851    | 0.866            | 0.928            |
| Advanced (+ color standardization) | 128        | SGD Nesterov | 0.25      | 0.907    | 0.826    | -                | -                |
| Advanced (+ color standardization) | 128        | Adam         | 0.3       | 0.922    | 0.853    | 0.867            | 0.929            |
| Advanced (+ color standardization) | 128        | Adam         | 0.2       | 0.922    | 0.853    | 0.870            | 0.930            |
| Advanced (+ color standardization + Blur) | 128        | Adam        |0.2        | 0.923    | 0.856      |
| Advanced (+ color standardization + Blur) | 128        | Adam        |0.25        | 0.919    | 0.847      |

## Run the solution

