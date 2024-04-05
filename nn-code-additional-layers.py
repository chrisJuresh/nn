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

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
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


transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 64

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
        self.dropout = nn.Dropout(0.5)
        
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
num_classes = 20
net = ModifiedCustomCNN(num_classes=num_classes).to(device)

# CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html START

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float().sum()
    accuracy = correct / y_true.shape[0]
    return accuracy

num_epochs=100

best_val_accuracy = 0.0

for epoch in range(num_epochs): 
    net.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    for inputs, labels in trainloader_augmented:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, labels).item()

    avg_train_loss = running_loss / len(trainloader_augmented)
    avg_train_accuracy = running_accuracy / len(trainloader_augmented)
    
    # Validation phase
    net.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, labels).item()

    avg_val_loss = val_loss / len(testloader)
    avg_val_accuracy = val_accuracy / len(testloader)
    
    # Checkpointing
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(net.state_dict(), 'best_model.pth')

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')
    
    lr_scheduler.step()

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