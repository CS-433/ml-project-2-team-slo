# -*- coding: utf-8 -*-
# -*- author : Yannis Laaroussi -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-12-03 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- Convolutional Network model -*-

import torch
import torch.nn as nn
import constants
from helpers import *

class CNN(nn.Module):
    def __init__(self,threshold=constants.FOREGROUND_THRESHOLD):

        self.threshold = threshold

        super(CNN, self).__init__()

    def train_model(self, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs=10):
        """
        Train the model.

        Args:
            optimizer (torch.optim): Optimizer used for training.
            scheduler (torch.optim.lr_scheduler): Scheduler used for training.
            criterion (torch.nn): Loss function.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader): Validation data loader.
            num_epochs (int): Number of epochs.
        """
        self.to(self.device)
        for epoch in range(num_epochs):
            self.train()
            for input, target in train_loader:
                input, target = input.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self(input)
                target = target.float().view(-1, 1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Validation loop
            self.eval()
            with torch.no_grad():
                total_correct = 0
                test_loss = 0
                tot_preds = []
                tot_targets = []
                for input, target in val_loader:
                    input, target = input.to(self.device), target.to(self.device)
                    output = self(input)
                    predictions = (output > self.threshold).float()
                    target = target.float().view(-1, 1)
                    total_correct += (predictions == target).sum().item()
                    test_loss += criterion(output, target).item() * len(input)
                    tot_preds.append(predictions)
                    tot_targets.append(target)

                test_loss /= len(val_loader.dataset)
                accuracy = total_correct / len(val_loader.dataset)
                f1 = f1_score(torch.cat(tot_preds).cpu().numpy(), torch.cat(tot_targets).cpu().numpy())
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {test_loss}, Validation Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}')

            scheduler.step(test_loss)

    def predict(self, test_loader):
        """
        Compute predictions on the test set.

        Args:
            test_loader (torch.utils.data.DataLoader): Test data loader.
        
        Returns:
            predictions (np.ndarray): Predictions on the test set.
        """
        self.eval()
        predictions = []
        device = next(self.parameters()).device

        with torch.no_grad():
            for input in test_loader:
                input = input.to(device)
                output = self(input)
                output = (output > self.threshold).float()
                predictions.append(output.cpu())

        return torch.cat(predictions).numpy().ravel()


class Advanced_CNN(CNN):
    def __init__(
            self,
            patch_size=constants.AUG_PATCH_SIZE,
            threshold = constants.FOREGROUND_THRESHOLD):
        """
        Constructor
        
        Layers:
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
        """
        self.threshold = threshold
        super(Advanced_CNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.1)
        self.relu4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu6 = nn.LeakyReLU(0.1)
        
        self.fc = nn.Linear(128 * (patch_size // 32) * (patch_size // 32), 1)

    def forward(self, x):
        """Forward pass."""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.dropout1(self.conv2(x))))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.dropout2(self.conv4(x))))
        x = self.pool5(self.relu5(self.conv5(x)))
        x = self.relu6(self.dropout3(self.conv6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Basic_CNN(CNN):
    def __init__(
            self,
            patch_size = constants.AUG_PATCH_SIZE,
            threshold = constants.FOREGROUND_THRESHOLD):
        """
        Constructor.
        
        Layers:
        - Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
        - ReLU activation
        - Max pooling layer with kernel size 2, stride 2
        - Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
        - ReLU activation
        - Fully connected layer with 1 output
        """
        self.threshold = threshold
        super(Basic_CNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        
        self.fc = nn.Linear(32 * (patch_size // 2) * (patch_size // 2), 1)


    def forward(self, x):
        """Forward pass."""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
