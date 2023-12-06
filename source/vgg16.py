# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-12-04 -*-
# -*- Last revision: 2023-12-04 (Vincent Roduit)-*-
# -*- python version : 3.11.6 -*-
# -*- VGG16 model -*-

#import files
import torch
import torch.nn as nn
from torchvision import models

#Import files
import constants
from helpers import f1_score

def vgg16(train_dataloader, validate_dataloader):
    # Load the pre-trained VGG16 model
    vgg16 = models.vgg16(pretrained=True)

    # Modify the last layer for your binary classification task
    in_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(in_features, 1)  # Assuming binary classification

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(vgg16.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1)

    # Training loop
    num_epochs = 5
    vgg16.to(constants.DEVICE)

    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(constants.DEVICE), labels.to(constants.DEVICE)

            optimizer.zero_grad()
            outputs = vgg16(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

        # Validation loop
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            # Validation loop
        vgg16.eval()
        # Disable gradient calculation for validation
        with torch.no_grad():
            total_correct = 0
            test_loss = 0
            tot_preds = []
            tot_targets = []
            for input, target in validate_dataloader:
                input, target = input.to(constants.DEVICE), target.to(constants.DEVICE)
                output = vgg16(input)
                predictions = (output > constants.FOREGROUND_THRESHOLD).float()
                target = target.float().view(-1, 1)
                total_correct += (predictions == target).sum().item()
                test_loss += criterion(output, target).item() * len(input)
                tot_preds.append(predictions)
                tot_targets.append(target)

            test_loss /= len(validate_dataloader)
            accuracy = total_correct / len(validate_dataloader)
            f1 = f1_score(torch.cat(tot_preds).cpu().numpy(), torch.cat(tot_targets).cpu().numpy())
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {test_loss}, Validation Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}')

            scheduler.step(test_loss)

        