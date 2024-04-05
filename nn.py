import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

tic = time.process_time()


# Custom cutout augmentation
class Cutout(object):
    """
    Randomly masks out one or more patches from an image.
    """

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

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


# Dataset augmentation transformation
transform_augmented = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Validation and testing dataset transformation
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Create dataloaders
batch_size = 64
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_augmented
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=12
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=12
)


class Block(nn.Module):
    """
    A block within the Backbone.

    Consists of:
        Linear layer.
        SpatialAveragePool.
        K Conv layers.
    """

    def __init__(self, in_channels, out_channels, num_convs, kernel_size=3):
        super(Block, self).__init__()
        # SpatialAveragePool to vector of d channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Linear layer transforming pooled output to a vector with K elements
        self.fc = nn.Linear(in_channels, num_convs)
        # K Conv layers
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size, padding=kernel_size // 2
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
                for _ in range(num_convs)
            ]
        )

    def forward(self, x):
        # Calculating weight vector a = [a1, ..., ak] with g
        weight_vector = F.softmax(self.fc(self.avg_pool(x).squeeze()), dim=-1)
        conv_outputs = torch.stack([conv(x) for conv in self.convs], dim=2)
        # Combining Conv layer outputs with calculated weights a to produce a single output O.
        weighted_output = torch.sum(
            conv_outputs * weight_vector.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), dim=2
        )
        return weighted_output


class Backbone(nn.Module):
    """
    Consists of N Blocks, each with a certain number of convolutional layers.
    """

    def __init__(
        self, num_blocks, num_convs, in_channels=3, out_channels_sequence=[32, 64, 128]
    ):
        super(Backbone, self).__init__()
        self.features = nn.Sequential()

        # Constructing Blocks and Pooling layers
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                self.features.add_module(
                    f"block_{i}_{j}",
                    Block(in_channels, out_channels_sequence[i], num_convs),
                )
                in_channels = out_channels_sequence[i]
            self.features.add_module(f"pool_{i}", nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    """
    Calculates the mean feature vector ƒ by applying SpatialAveragePool to the Backbone's output. Then passes ƒ through a MLP classifier.
    """

    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        # Computes a mean feature vector ƒ by applying SpatialAveragePool to the Backbone's output.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_features = in_features * 2

        # MLP classifier
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Model(nn.Module):
    """
    Architecture based on Convolutional Neural Networks that uses a Backbone and a Classifier.
    """

    def __init__(self, num_classes, num_blocks=[2, 2, 2], num_convs=3):
        super(Model, self).__init__()
        out_channels_sequence = [32, 64, 128]

        # Backbone
        self.backbone = Backbone(
            num_blocks,
            num_convs,
            in_channels=3,
            out_channels_sequence=out_channels_sequence,
        )

        # Classifier
        final_in_channels = out_channels_sequence[-1]
        self.classifier = Classifier(final_in_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        # Process the output of the last Block in the Backbone through Classifier
        x = self.classifier(x)
        return x


# Calculate accuracy helper
def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float().sum()
    accuracy = correct / y_true.shape[0]
    return accuracy


# Calculate training loss and accuracy
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, running_accuracy = 0.0, 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, labels).item()

    avg_loss = running_loss / len(loader)
    avg_accuracy = running_accuracy / len(loader)
    return avg_loss, avg_accuracy


# Evaluate the model on the test set
def evaluate_model(model, loader, criterion, device):
    model.eval()
    test_loss, test_accuracy = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(outputs, labels).item()

    avg_loss = test_loss / len(loader)
    avg_accuracy = test_accuracy / len(loader)
    return avg_loss, avg_accuracy


# Initialize lists to store metrics
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

# Use GPU
device = torch.device("cuda:0")

# Initialize model
num_classes = 20
net = Model(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Initialise training variables
num_epochs = 50

# Training and testing loop
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_epoch(
        net, trainloader, optimizer, criterion, device
    )
    lr_scheduler.step()

    test_loss, test_accuracy = evaluate_model(net, testloader, criterion, device)

    # Store metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot training and testing losses
axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(test_losses, label="Test Loss")
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Plot training and testing accuracies
axes[1].plot(train_accuracies, label="Train Accuracy")
axes[1].plot(test_accuracies, label="Test Accuracy")
axes[1].set_title("Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()

#### CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html START ####
# Test the final model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Final model accuracy: {100 * correct / total:.4f} %")
#### CODE FROM https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html END ####

toc = time.process_time()
print("Time Taken (s):", toc - tic)
