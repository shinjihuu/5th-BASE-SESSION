### X:AI 1주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG16

batch_size = 100
learning_rate = 0.0002
num_epoch = 50

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform > 이미지 텐서로 변환 및 정규화
transform = transforms.Compose( 
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 datasets 불러오기
cifar10_train = datasets.CIFAR10(root="./Data/", train=True, transform=transform, target_transform=None, download=True)
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# VGG16 인스턴스 생성
model = VGG16(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_lst = []
for i in range(num_epoch): 
    for k,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_= label.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(f'{i} epoch loss : {loss}')
        loss_lst.append(loss.cpu().detach().numpy())


    # 학습 후 모델 저장
    torch.save(model.state_dict(), './VGG16_cifar10.pth')