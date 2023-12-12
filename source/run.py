# -*- coding: utf-8 -*-
# -*- author : Yannis Laaroussi -*-
# -*- date : 2023-10-29 -*-
# -*- Last revision: 2023-12-08 -*-
# -*- python version : 3.11.5 -*-
# -*- Description: Run the best solution -*-

import time
import argparse
import os
import re
import torch
import torch.nn as nn

from test_data import TestData
from helpers import *
from cnn import Advanced_CNN, Basic_CNN
from data_processing import AdvancedProcessing
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define the models that need to be standardized
is_standardized = {
    'basic_cnn_16.pth': False,
    'basic_cnn_32.pth': False,
    'basic_cnn_64.pth': False,
    'advanced_cnn_64.pth': False,
    'advanced_cnn_128.pth': False,
    'advanced_cnn_color_128.pth': True,
    'advanced_cnn_128_adamw.pth': True,
    'advanced_cnn_128_nesterov.pth': True,
    'advanced_cnn_128_thr_02.pth': True,
    'advanced_cnn_128_thr_03.pth': True,
    'advanced_cnn_128_blur_025.pth': True,
    'advanced_cnn_128_blur.pth': True

}

def handle_path(data_path, csv_path, mask_path, model_path):
    """
    Handle the paths, take the default ones if not provided or
    raise an error if the path does not exist.

    Args:
        data_path (str): Path to the data.
        csv_path (str): Path to the csv file.
        mask_path (str): Path to the mask file.
        model_path (str): Path to the model.
    
    Returns:
        data_path (str): Path to the data.
        csv_path (str): Path to the csv file.
        mask_path (str): Path to the mask file.
        model_path (str): Path to the model.
    """
    # Handle the paths
    if data_path is None:
        # If no path is provided, use the default one
        data_path = constants.ROOT_DIR
    elif not os.path.exists(data_path):
        # If the path does not exist, raise an error
        raise FileNotFoundError(f"Path {data_path} does not exist.")
    else:
        if model_path is not None:
            if 'test_set_images' not in os.listdir(data_path):
                # If the path does not contain the test folder, raise an error
                raise FileNotFoundError(f"Path {data_path} does not contain the test folder.")
        else: 
            if 'test_set_images' not in os.listdir(data_path) and 'training' not in os.listdir(data_path):
                # If the path does not contain the test and training folders, raise an error
                raise FileNotFoundError(f"Path {data_path} does not contain the test and training folders.")

    if csv_path is None:
        # If no path is provided, use the default one
        csv_path = os.path.join(constants.SUBMISSION_PATH, 'submission.csv')
    elif not os.path.exists(csv_path):
        # If the path does not exist, raise an error
        raise FileNotFoundError(f"Path {csv_path} does not exist.")
    else:
        # If the path is a directory, add the default filename
        if os.path.isdir(csv_path):
            csv_path = os.path.join(csv_path, 'submission.csv')
    
    if mask_path is None:
        # If no path is provided, use the default one
        mask_path = os.path.join(constants.RESULTS_FOLDER_PATH, 'masks')
    elif not os.path.exists(mask_path):
        # If the path does not exist, raise an error
        raise FileNotFoundError(f"Path {mask_path} does not exist.")
    
    if model_path is not None:
        if not os.path.exists(model_path):
            # If the path does not exist, raise an error
            raise FileNotFoundError(f"Path {model_path} does not exist.")
        else:
            if not os.path.isfile(model_path):
                # If the path is not a file, raise an error
                raise FileNotFoundError(f"Path {model_path} is not a file.")
            elif not os.path.basename(model_path).endswith('.pth'):
                # If the file is not a .pth file, raise an error
                raise FileNotFoundError(f"File {model_path} is not a .pth file.")
    
    return data_path, csv_path, mask_path, model_path

def run(data_path, csv_path, mask_path, model_path):
    """
    Run the best solution.
    
    Args:
        data_path (str): Path to the data.
        csv_path (str): Path to the csv file.
        mask_path (str): Path to the mask file.
        model_path (str): Path to the model.
    """
    # Set the seed
    torch.manual_seed(0)
    # Handle the paths
    data_path, csv_path, mask_path, model_path = handle_path(data_path, csv_path, mask_path, model_path)

    if model_path is None:
        # If no model path is provided, train the best model from start
        # Load the train images
        patch_size = 128
        standardize = True
        myDatas = AdvancedProcessing(standardize=standardize, aug_patch_size=patch_size, blur=True)
        myDatas.proceed()
        
        # Define the model
        cnn = Advanced_CNN(patch_size, threshold=0.2)

        # Define the criterion
        criterion = nn.BCEWithLogitsLoss()
        # Define the optimizer
        optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
        # Define the scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        # Train the model
        cnn.train_model(
            optimizer,
            scheduler,
            criterion,
            myDatas.train_dataloader,
            myDatas.validate_dataloader,
            num_epochs=20)
    else:
        # If a model path is provided, load the model
        # Get the patch size from the model name
        pattern = r'\d+'
        model_filename = os.path.basename(model_path)
        patch_size = int(re.search(pattern, model_filename).group(0))
        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn = torch.load(model_path, map_location=device)
        standardize = is_standardized[model_filename]

    # Load the test images
    myTestDatas = TestData(standardize=standardize, aug_patch_size=patch_size)
    myTestDatas.proceed(test_path=os.path.join(data_path, 'test_set_images'))

    # Predict the test images
    print("Predicting...")
    preds = cnn.predict(myTestDatas.test_dataloader)
    print("Done!")
    
    # Save the predictions as PNG
    images_filenames = save_pred_as_png(preds, len(myTestDatas.imgs), folder_path=mask_path)
    # Create the submission file
    masks_to_submission(csv_path, *images_filenames)


if __name__ == "__main__":
    # initialiaze an argument parser
    parser = argparse.ArgumentParser("Path to the dataset and the output file")
    parser.add_argument(
        "--data_path", type=str, help="path to the dataset", required=False
    )
    parser.add_argument(
        "--output_csv_path", type=str, help="submission file path", required=False
    )
    parser.add_argument(
        "--output_mask_path", type=str, help="output masks image path", required=False
    )
    parser.add_argument(
        "--model_path", type=str, help="path to the model", required=False
    )
    args = parser.parse_args()

    # get the value of the data path
    data_path = args.data_path
    output_csv_path = args.output_csv_path
    output_mask_path = args.output_mask_path
    model_path = args.model_path

    # get the start time
    st = time.time()

    # run script
    run(data_path, output_csv_path, output_mask_path, model_path)

    # get the end time
    et = time.time()
    # get the execution time in minutes
    elapsed_time = et - st
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"Process finished in {int(minutes)} min {int(seconds)} sec")