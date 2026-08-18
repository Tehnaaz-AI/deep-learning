# Write a programme to build an image classifier using Pytorch and CNN on cifar dataset

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms


# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize( (0.5, 0.5, 0.5) , (0.5, 0.5, 0.5) ),
])

# Download Data
train_data = CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transform,
)

test_data = CIFAR10(
    root = './data',
    train = False,
    download = True,
    transform = transform,
)

# Loading Data 

train_loader = DataLoader(
    dataset = train_data,
    batch_size = 32,
    shuffle = True,
)

test_loader = DataLoader(
    dataset = test_data,
    batch_size = 32,
    shuffle = False,
)

# Build the Architecture

class cifar10_CNN(nn.Module):
    def __init__(self):
        super(cifar10_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, 
                               out_channels = 16, 
                               kernel_size = 5, 
                               )
        self.pool = nn.MaxPool2d(kernel_size = 2,
                                  stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, 
                               out_channels = 16, 
                               kernel_size = 5, 
                               )
        self.pool = nn.MaxPool2d(kernel_size = 2,
                                  stride = 2)

        self.fc1 = nn.Linear(in_features = 5*5*16, 
                             out_features = 256)
        self.fc2 = nn.Linear(in_features = 256,
                             out_features = 10)

    # forward
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        
        
model = cifar10_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

for epoch in range(10):

    for image, label in train_loader:
        predicted = model(image)
        loss = criterion(predicted, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.item()}")

    # model evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in test_loader:
            predicted = model(image)
            max_prob , predicted_class = torch.max(predicted.data, 1)
            total += label.size(0)
            correct += (predicted_class == label).sum().item()

    Accuracy = (correct/total) * 100
    print(f"Accuracy: {Accuracy}%")




print("Hii Jarvis")



