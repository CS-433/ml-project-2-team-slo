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
- [Dataset Structure](#dataset-structure)
- [Data Wrangling](#data-wrangling)
- [Models](#models)
- [Results](#results)
- [Run the solution](#run-the-solution)

## Abstract 
The purpose of this project is to create a binary classifier that is able to recognize roads from Google maps satellite images. This kind of task is very common in image classification and computer vision. This repository presents solutions addressed to solve this problem. The major issues that have to be handled are the following:
* The datas are not well balanced : only 25% of the datas are roads.
* Roads are majoritary vertical or horizontal.
* The color of the roads is very similar to the one of sidewalk or parking area.

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
The folder `source` provides all the codes used to develop the models. All the tested models are grouped in the folder `models` and can be reused without training (see the section **Run the solution** for more help. Folder `template` provides the given codes to start the projects. Some of the functions introduced in this code has been use in the code.

## Dataset Structure

```
.
├── test_set_images
│   ├── test_1
│   │   └── test_1.png
│   ├── ...
│   │   └── ...
│   ├── test_50
│   │   └── test_50.png
└── training
    ├── groundtruth
    │   ├── satImage_001.png
    │   ├── ...
    │   └── satImage_100.png
    └── images
        ├── satImage_001.png
        ├── ...
        └── satImage_100.png
```

## Data wrangling
The training datas consist of a set of 100 RGB images of size 400x400 pixels, comming with the correspond label images. The first step is to convert the images to arrays that can be used later on. The predictions are done on patches of size 16x16 pixels. Therefore groundtruth images have to be croped to paches of this size.

The test images consist of 50 RGB images of size 608x608 pixels. These images are also croped into patches of size 16x16 in order to make predictions in the plateform [AICrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)

## Data processing
To classes are done to perform data processing: `BasicProcessing``
* load datas
* create patches : crop the images into patches of size 16x16
* create labels : create the labels of the patches
* create sets: create a validation from the original set to evaluate the model

This model doesn't address the problems mentionned in the abstract. The purpose of this class was only a first step to understand the datas and be able to produce results. 

The second class, `AdvancedProcessing`, proposes solutions to problems cite in introduction. It performs the following transformations:
* load datas
* standardize color
* split sets: split images into training and validate sets
* create patches: create samples of size = augmented patch size (**for the training only**) with the following transformations:
    * Rotations
    * Bluring
    * Random center
    * Balanced number of road and background

It's good noticing that only samples for the training are balanced. Patches of the validation set is create in the "standard" fashion. Therefore, the obtained F1-score for the validation is done on unbalanced datas.

## Models
Two models are proposed for this project, a basic and an advanced convolution neural network. The structure of the models is defined as follow:

**Basic CNN**
- Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
- ReLU activation
- Max pooling layer with kernel size 2, stride 2
- Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
- ReLU activation
- Fully connected layer with 1 output

**Advanced CNN**
- Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
- ReLU activation
- Max pooling layer with kernel size 2, stride 2
- Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
- Dropout layer with probability 0.1
- ReLU activation
- Max pooling layer with kernel size 2, stride 2
- Convolutional layer with 64 filters, kernel size 3, stride 1, padding 1
- ReLU activation
- Max pooling layer with kernel size 2, stride 2
- Convolutional layer with 64 filters, kernel size 3, stride 1, padding 1
- Dropout layer with probability 0.1
- ReLU activation
- Max pooling layer with kernel size 2, stride 2
- Convolutional layer with 128 filters, kernel size 3, stride 1, padding 1
- ReLU activation
- Max pooling layer with kernel size 2, stride 2
- Convolutional layer with 128 filters, kernel size 3, stride 1, padding 1
- Dropout layer with probability 0.1
- ReLU activation
- Fully connected layer with 1 output
- 
## Results

| Model                                      | Patch size | Optimizer    | Threshold | Accuracy | F1-score | AICrowd F1-Score | AICrowd accuracy |
|--------------------------------------------|------------|--------------|-----------|----------|----------|------------------|------------------|
| Basic                                      | 16         | Adam         | 0.25      | 0.800    | 0.679    | -                | -                |
| Basic                                      | 32         | Adam         | 0.25      | 0.832    | 0.720    | -                | -                |
| Basic                                      | 64         | Adam         | 0.25      | 0.850    | 0.743    | 0.783            | 0.876            |
| Advanced                                   | 64         | Adam         | 0.25      | 0.889    | 0.792    | -                | -                |
| Advanced                                   | 128        | Adam         | 0.25      | 0.912    | 0.833    | 0.857            | 0.923            |    
| Advanced (+ color standardization)         | 128        | Adam         | 0.25      | 0.924    | 0.857    | 0.860            | 0.926            |
| Advanced (+ color standardization)         | 128        | AdamW        | 0.25      | 0.921    | 0.851    | -                | -                |
| Advanced (+ color standardization)         | 128        | SGD Nesterov | 0.25      | 0.907    | 0.826    | -                | -                |
| Advanced (+ color standardization)         | 128        | Adam         | 0.3       | 0.922    | 0.853    | -                | -                |
| Advanced (+ color standardization)         | 128        | Adam         | 0.2       | 0.922    | 0.853    | 0.866            | 0.928            |
| Advanced (+ color standardization + Blur)  | 128        | Adam        |0.2         | 0.923    | 0.856    | 0.870 	          | 0.930            |
| Advanced (+ color standardization + Blur)  | 128        | Adam        |0.25        | 0.919    | 0.847    | 0.861            | 0.926            |

## Run the solution


The file run.py enables to directly load one of the different trained model present in the folder '/models' and make the predictions of the test set or to train the best model from the beginning. The following arguments can be passed:

- `--data_path`: path to the dataset folder which should follow structure described in section [Dataset Structure](#dataset-structure).
- `--output_csv_path`: path to the directory where the submission.csv will be written.
- `--output_mask_path`: path to the directory where the predicted mask images are saved.
- `--model_path`: path to the model file to load (it will train and predict the best model if not specified).
