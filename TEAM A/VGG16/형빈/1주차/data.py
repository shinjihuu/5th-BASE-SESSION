from vgg16 import *

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

batch_size = 32
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
train = datasets.CIFAR10(root = "./data",train=True,transform=transform,download=True)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                        shuffle=True)
test = datasets.CIFAR10(root = "./data",train=False,transform=transform,download=True)
test_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                        shuffle=True)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')