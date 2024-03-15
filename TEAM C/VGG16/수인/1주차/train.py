import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG

# Hyperparameters
batch_size = 100
learning_rate = 0.0002
num_epoch = 100

## model, loss, optimizer 선언
# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 클래스를 인스턴스화
model = VGG(base_dim=64).to(device)

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss() # (CrossEntropyLoss: 분류 문제에 적합)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## TRAIN/TEST 데이터셋 정의
# Transform 정의
transform = transforms.Compose( # (데이터 정규화)
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 TRAIN 데이터 정의
cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, target_transform=None, download=True)
train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

loss_arr = []
for i in range(num_epoch): 
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_= label.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()

    if i % 10 ==0:
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())
        # detach: 그래디언트 계산을 중단하고 텐서를 분리하는 역할 