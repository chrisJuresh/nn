import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

tic = time.process_time() 

# CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html START

transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomCrop(32, padding=4),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 32

trainset_augmented = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_augmented)
trainloader_augmented = torch.utils.data.DataLoader(trainset_augmented, batch_size=batch_size,
                                          shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=12)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html END

class DynamicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, kernel_size=3):
        super(DynamicConvBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_convs)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for _ in range(num_convs)
        ])
        
    def forward(self, x):
        weight_vector = F.softmax(self.fc(self.avg_pool(x).squeeze()), dim=-1)
        conv_outputs = torch.stack([conv(x) for conv in self.convs], dim=2)  # Shape: [N, C_out, num_convs, H, W]
        weighted_output = torch.sum(conv_outputs * weight_vector.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), dim=2)
        return weighted_output

class ClassifierBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ClassifierBlock, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.dense(x))
        return self.dropout(x)

class ModifiedCustomCNN(nn.Module):
    def __init__(self, num_classes, num_blocks=[2, 2, 2], num_convs=3):
        super(ModifiedCustomCNN, self).__init__()
        in_channels = 3
        out_channels_sequence = [32, 64, 128]
        layers = []
        
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                layers.append(DynamicConvBlock(in_channels, out_channels_sequence[i], num_convs))
                in_channels = out_channels_sequence[i]
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            ClassifierBlock(128 * 4 * 4, 1024),  # Adjust the flattening size accordingly
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# Initialize the model
num_classes = 10
net = ModifiedCustomCNN(num_classes=num_classes).to(device)

# CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html START

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float().sum()
    accuracy = correct / y_true.shape[0]
    return accuracy

num_epochs=100

for epoch in range(num_epochs): 
    running_loss = 0.0
    running_accuracy = 0.0
    for i, data in enumerate(trainloader_augmented, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, labels).item()

        if i % 200 == 199:  # Print every 200 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader_augmented)}], Loss: {running_loss / 200:.4f}, Accuracy: {running_accuracy / 200:.4f}')
            running_loss = 0.0
            running_accuracy = 0.0

print('Finished Training')

# Test loop (make sure to replace 'testloader' with your DataLoader)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)  # Move data to the device
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# Save the trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html END

toc = time.process_time() 
print(toc - tic)
