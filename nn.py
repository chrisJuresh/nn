from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time


# Code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNNClassifier(nn.Module):
    def __init__(self):
        super(CIFAR10CNNClassifier, self).__init__()
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # 10 classes in CIFAR-10
        )

    def forward(self, x):
        # Apply feature extraction
        x = self.feature_extractor(x)
        # Flatten the output for the classifier
        x = x.view(-1, 128 * 4 * 4)
        # Apply classifier
        x = self.classifier(x)
        return x

# Instantiate the model
model = CIFAR10CNNClassifier()
print(model)


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# Data transformation and normalization
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-10 datasets and dataloaders
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Model instantiation
model = CIFAR10CNNClassifier()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

epoch_losses = []

for epoch in range(10):  # Number of epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(trainloader)
    epoch_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

print('Finished Training')

import matplotlib.pyplot as plt

# Plotting the training loss
plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random test images
dataiter = iter(testloader)
images, labels = dataiter.next()

# Show images
imshow(torchvision.utils.make_grid(images))

# Predict labels
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# Print predicted and actual labels
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
print('Actual: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# Save the trained model
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)
