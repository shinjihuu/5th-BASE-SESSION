import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG
from PIL import Image


# Hyperparameters
learning_rate = 0.00005
num_epoch = 50
batch_size = 10

## model, loss, optimizer 선언
# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 클래스를 인스턴스화
model = VGG(base_dim=64).to(device)

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss() # (CrossEntropyLoss: 분류 문제에 적합)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam 옵티마이저 사용

path = './data/' # 데이터 경로

num_classes = 4 # 클래스 개수
IMG_SIZE = (32, 32)   # resize image

## TRAIN/TEST 데이터셋 정의
# Transform 정의
transform = transforms.Compose( # (데이터 정규화)
    [transforms.Resize(IMG_SIZE),
     transforms.ToTensor(), # 이미지를 텐서로 변환
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transforms=None, target_transform=None):
        super(MyDataset, self).__init__()
        with open(root + datatxt, 'r') as f:
            imgs = []
            for line in f:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
            self.root = root
            self.imgs = imgs
            self.transforms = transforms
            self.target_transform = target_transform

    def __getitem__(self, index):
        f, label = self.imgs[index]  
        img = Image.open(self.root + f).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img) # 이미지 전처리 수행
        return img, label

    def __len__(self):
        return len(self.imgs)

# CIFAR10 TRAIN 데이터 정의
train_data = MyDataset(path + 'train_data/', 'train_label.txt', transforms=transform) # 데이터셋 인스턴스화
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

loss_arr = [] # 손실 기록을 위한 리스트
for i in range(num_epoch): 
    for j,[image,label] in enumerate(train_loader): # 각 미니배치에 대해 반복
        x = image.to(device) # 이미지를 지정한 장치로 이동
        y_= label.to(device) # 레이블을 지정한 장치로 이동
        
        optimizer.zero_grad() # 그래디언트 초기화
        output = model.forward(x) # 모델에 이미지를 전달하여 예측
        loss = loss_func(output,y_) # 손실 계산
        loss.backward() # 역전파를 통해 그래디언트 계산
        optimizer.step() # 옵티마이저로 모델 파라미터 업데이트

    if i % 10 ==0:
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())
        # detach: 그래디언트 계산을 중단하고 텐서를 분리하는 역할 

    torch.save(model.state_dict(), "./VGG16_newdata.pth") #학습 완료된 모델 저장