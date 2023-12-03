import torch
import torch.nn as nn
from constants import *
from helpers import *

class Basic_CNN(nn.Module):
    def __init__(self, patch_size):
        """Constructor."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(Basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

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

    
    def train_model(self, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs=10):
        """Train the model."""
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
                
            # Step the scheduler after each epoch
            scheduler.step()
            # Validation loop
            self.eval()
            # Disable gradient calculation for validation
            with torch.no_grad():
                total_correct = 0
                test_loss = 0
                tot_preds = []
                tot_targets = []
                for input, target in val_loader:
                    input, target = input.to(self.device), target.to(self.device)
                    output = self(input)
                    predictions = (output > FOREGROUND_THRESHOLD).float()
                    target = target.float().view(-1, 1)
                    total_correct += (predictions == target).sum().item()
                    test_loss += criterion(output, target).item() * len(input)
                    tot_preds.append(predictions)
                    tot_targets.append(target)

                test_loss /= len(val_loader.dataset)
                accuracy = total_correct / len(val_loader.dataset)
                f1 = f1_score(torch.cat(tot_preds).cpu().numpy(), torch.cat(tot_targets).cpu().numpy())
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {test_loss}, Validation Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}')


    def predict(self, test_loader):
      """Predict the labels of the test set."""
      self.eval()
      predictions = []
      device = next(self.parameters()).device  # Get the device of the model (assuming all parameters are on the same device)

      with torch.no_grad():
          for input in test_loader:
              input = input.to(device)  # Move input data to the same device as the model
              output = self(input)
              output = (output > FOREGROUND_THRESHOLD).float()
              predictions.append(output.cpu())  # Move predictions back to CPU

      return torch.cat(predictions).numpy().ravel()

