import torch
import torch.nn as nn
from constants import *
from helpers import *
import matplotlib.pyplot as plt

class Basic_CNN(nn.Module):
    def __init__(self, patch_size):
        super(Basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(128 * (patch_size // 8) * (patch_size // 8), 128)
        # self.fc2 = nn.Linear(128, 1)
        self.fc1 = nn.Linear(64 * (patch_size // 4) * (patch_size // 4), 64)
        self.fc2 = nn.Linear(64, 1)
        

    def forward(self, x):
        """Forward pass."""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

   
    def train_model(self, optimizer, criterion, train_loader, val_loader, num_epochs=10):
        losses = []
        F1_score = []
        acc = []
        for epoch in range(num_epochs):
            self.train()
            for input, target in train_loader:
                optimizer.zero_grad()
                output = self(input)
                target = target.float().view(-1, 1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
            # Validation loop
            self.eval()
            # Disable gradient calculation for validation
            with torch.no_grad():
                total_correct = 0
                test_loss = 0
                tot_preds = []
                tot_targets = []
                for input, target in val_loader:
                    output = self(input)
                    predictions = (output > FOREGROUND_THRESHOLD).float()
                    target = target.float().view(-1, 1)
                    total_correct += (predictions == target).sum().item()
                    test_loss += criterion(output, target).item() * len(input)
                    tot_preds.append(predictions)
                    tot_targets.append(target)

                test_loss /= len(val_loader.dataset)
                accuracy = total_correct / len(val_loader.dataset)
                f1 = f1_score(torch.cat(tot_preds).numpy(), torch.cat(tot_targets).numpy())
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {test_loss}, Validation Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}')
                losses.append(test_loss)
                F1_score.append(f1)
                acc.append(accuracy)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(range(num_epochs),losses)
        axes[0].set_title('Loss')
        axes[1].plot(range(num_epochs),F1_score)
        axes[1].set_title('F1 score')
        axes[2].plot(range(num_epochs),acc)
        axes[2].set_title('Accuracy')
        plt.show()




    def predict(self, test_loader):
        """Predict the labels of the test set."""
        self.eval()
        predictions = []
        with torch.no_grad():
            for input in test_loader:
                output = self(input)
                output = (output > FOREGROUND_THRESHOLD).float()
                predictions.append(output)
        return torch.cat(predictions).numpy()
    


    


        

