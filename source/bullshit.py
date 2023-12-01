import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19

import constants
from preprocessing_helper import *
from postprocessing_helper import *


class DeepModel(nn.Module):
    """ Our best performing model inspired by the VGG architecture """
    
    WINDOW_SIZE = 100
    OUTPUT_FILENAME = "deep_model"

    def __init__(self):
        super(DeepModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (self.WINDOW_SIZE // 16) * (self.WINDOW_SIZE // 16), 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model, train_loader, val_loader, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                val_loss += criterion(output, labels).item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {(correct/total)*100:.2f}%")


def generate_images(model, imgs, gt_imgs):
    prediction_training_dir = "predictions_training/"
    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)
    
    for i in range(1, constants.TRAINING_SIZE+1):
        pimg = get_prediction_with_mask(imgs[i-1], model, DeepModel.WINDOW_SIZE)
        Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        oimg = get_prediction_with_overlay(imgs[i-1], model, DeepModel.WINDOW_SIZE)
        oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
    
    checkImageTrainSet(model, imgs, gt_imgs, DeepModel.WINDOW_SIZE)


def generate_submission(model):
    createSubmission(model, DeepModel.WINDOW_SIZE)


def save(model):
    torch.save(model.state_dict(), DeepModel.OUTPUT_FILENAME + ".pth")
    print("Saved model to disk")


def load(model, weights_dir):
    model.load_state_dict(torch.load(weights_dir))
    print("Loaded model from disk")


def main():
    model = DeepModel()

    image_dir = "path/to/image/directory"
    gt_dir = "path/to/ground_truth/directory"
    training_size = constants.TRAINING_SIZE

    dataset = ImageFolder(image_dir, transform=ToTensor())
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model.load_data(image_dir, gt_dir, training_size)
    train(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
