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
num_epoch = 100

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform > 이미지 텐서로 변환 및 정규화
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 datasets 불러오기
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

test_loader = DataLoader(cifar10_test, batch_size = batch_size, shuffle=False)


# VGG16 인스턴스 생성
model = VGG16(base_dim=64).to(device)
# pre trained model load
model.load_state_dict(torch.load('./VGG16_cifar10.pth'))

correct = 0 # correct count
total = 0 # total count

model.eval()

# 가중치 갱신 X
with torch.no_grad():
    for i, [image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        _, output_index = torch.max(output,1) 

        total += label.size(0)
        correct += (output_index==y).sum().float()
    
    print("Accuracy of Test Data: {}%".format(100*correct/total))
    