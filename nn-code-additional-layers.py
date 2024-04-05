import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

tic = time.process_time() 

# Custom cutout augmentation
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

# Dataset augmentation transformation
transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Unprocessed dataset transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Get datasets
trainset_augmented = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split datasets
train_size = int(0.9 * len(trainset_augmented))
val_size = len(trainset_augmented) - train_size
indices = list(range(len(trainset_augmented)))
train_indices, val_indices = indices[:train_size], indices[train_size:]
val_dataset_non_augmented = Subset(trainset, val_indices)

# Load datasets
batch_size = 64
trainloader = DataLoader(Subset(trainset_augmented, train_indices), batch_size=batch_size, shuffle=True, num_workers=12)
valloader = DataLoader(val_dataset_non_augmented, batch_size=4, shuffle=False, num_workers=12)

# Model
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, kernel_size=3):
        super(Block, self).__init__()
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
        conv_outputs = torch.stack([conv(x) for conv in self.convs], dim=2) 
        weighted_output = torch.sum(conv_outputs * weight_vector.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), dim=2)
        return weighted_output

class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.dense(x))
        return self.dropout(x)

class CNN(nn.Module):
    def __init__(self, num_classes, num_blocks=[2, 2, 2], num_convs=3):
        super(CNN, self).__init__()
        in_channels = 3
        out_channels_sequence = [32, 64, 128]
        layers = []
        
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                layers.append(Block(in_channels, out_channels_sequence[i], num_convs))
                in_channels = out_channels_sequence[i]
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Backbone
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            Classifier(128 * 4 * 4, 1024), 
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# Use GPU
device = torch.device('cuda:0')

# Initialize model
num_classes = 20
net = CNN(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float().sum()
    accuracy = correct / y_true.shape[0]
    return accuracy

# Initialise training variables
num_epochs=50
best_val_accuracy = 0.0

# Train the network
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, labels).item()

    avg_train_loss = running_loss / len(trainloader)
    avg_train_accuracy = running_accuracy / len(trainloader)
    
    # Validation
    net.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, labels).item()

    avg_val_loss = val_loss / len(valloader)
    avg_val_accuracy = val_accuracy / len(valloader)
    
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(net.state_dict(), 'best_model.pth')

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')
    
    lr_scheduler.step()

print('Finished Training')

# Test the network
#### CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html START ####
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=12)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# Save the model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
#### CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html END ####

toc = time.process_time() 
print('Time (s):', toc - tic)
