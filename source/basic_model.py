# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2023-11-25 -*-
# -*- Last revision: 2023-11-25 -*-
# -*- python version : 3.11.6 -*-
# -*- Description: Basic model implementation-*-

#import libraiies
import torch

#Define the model
class LeNet(torch.nn.Module):
    def __init__(self):
        """Initialize the network"""
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        """Forward path"""
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    def fit(self, X,y):
        """Fit datas to the model"""
        self.X_train = X
        self.y_train = y

    def accuracy(predicted_logits, reference):
        """
        Compute the ratio of correctly predicted labels
        """
        labels = torch.argmax(predicted_logits, 1)
        correct_predictions = labels.eq(reference)
        return correct_predictions.sum().float() / correct_predictions.nelement()

    def train(self,criterion, X_test, y_test, optimizer, num_epochs,device):
        """Train de model"""
        for epoch in range(num_epochs):
            self.X_train, self.y_train = self.X_train.to(device), self.y_train.to(device)
            prediction = self.forward(self.X_train)
            loss = criterion(prediction, self.y_train)

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()
            LeNet.eval()
            accuracies_test = []
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Evaluate the network (forward pass)
            prediction = self.forward(X_test)
            accuracies_test.append(self.accuracy(prediction, y_test))

            print(
                "Epoch {} | Test accuracy: {:.5f}".format(
                    epoch, sum(accuracies_test).item() / len(accuracies_test)
                )
            )
