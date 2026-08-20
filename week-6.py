# Write a programme to build an image classifier using Pytorch
# and CNN based pretrained model (resnet18) on custom image dataset (ants and bees)

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import transforms

# 124,120,74,80

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize( (0.5) , (0.5) ),
])

# Download Data
train_data = ImageFolder(
    root = './data/ants-bees/hymenoptera_data/train',
    transform = transform,
)

test_data = ImageFolder(
    root = './data/ants-bees/hymenoptera_data/val',
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

        
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print(model)
"""


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



"""
print("Hii Jarvis")



