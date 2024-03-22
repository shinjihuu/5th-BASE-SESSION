#import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
from VGG16 import VGG16 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cuda gpu사용

#hyperparameter
batch_size = 100
learning_rate = 0.0002
num_epoch = 30

normalize = transforms.Normalize( # 정규화
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)#색상 rgb normalize

# define transforms: 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize, # 정규화
])

# CIFAR10 데이터셋 load
cifar10_train = datasets.CIFAR10(root='./Data/',train=True,
                                 transform=transform,target_transform=None,
                                 download=True) # train 데이터셋
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, 
                                transform=transform, target_transform=None, 
                                download=True) # test 데이터셋

## DataLoader: 데이터를 배치 형태로 로드, 
### 지정한 배치 크기에 따라 데이터 자동으로 나누어 줌
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)


# 모델 초기화
model = VGG16(base_dim=64).to(device)

# Loss and optimizer
loss_func = nn.CrossEntropyLoss() #손실함수
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 최적화 알고리즘

# Train
loss_arr = []
total_step = len(train_loader)

for epoch in range(num_epoch): 
    for i, (image,label) in enumerate(train_loader):
        # Move tensors to the configured device
        image = image.to(device)
        labels = label.to(device)

        # Forward pass
        optimizer.zero_grad() #이전 grad 초기화
        outputs = model.forward(image) #모델 예측값
        
        loss = loss_func(outputs,labels) #손실함수 계산
        loss.backward()# 역전파로 grad다시 계산
        optimizer.step() # 최적화, 파라미터 업데이트

    if i%10 ==0:
        # Epoch별 손실 출력 
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epoch, i+1, total_step, loss.item()))
        loss_arr.append(loss.cpu().detach().numpy()) 

torch.save(model.state_dict(), "./train_model/VGG16_01.pth")

