import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm

from unet import UNet

data_dir = os.path.join("datasets/cityscapes_data")
train_dir = os.path.join(data_dir, "train") 
val_dir = os.path.join(data_dir, "val")

train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)


#label 값 정의
num_items = 1000
#0~255까지 숫자 중, 무작위로 선택된 3*1000 개의 숫자 배열 생성
color_array = np.random.choice(range(256), 3*num_items).reshape(-1,3)

#임의로 랜덤 색상 뽑아낸 것들을 클래스 10개로 군집화 진행
num_classes = 10
label_model = KMeans(n_clusters = num_classes)
label_model.fit(color_array)

#datasets 정의
class CityscapeDataset(Dataset):
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.label_model = label_model

    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert("RGB")
        image = np.array(image)
        cityscape, label = self.split_image(image)
        # KMeans 모델인 label_model을 사용하여 입력된 픽셀 색상 정보를 바탕으로 각 픽셀에 대한 클래스(레이블)를 예측
        #.reshape(256,256): 예측된 클래스를 원래 이미지의 모양으로 다시 재구성
        label_class = self.label_model.predict(label.reshape(-1,3)).reshape(256,256)
        label_class = torch.Tensor(label_class).long()
        cityscape = self.transform(cityscape)
        return cityscape, label_class

    #source, label 나누기
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

batch_size = 16
epochs = 10
lr = 0.01

dataset = CityscapeDataset(train_dir, label_model)
data_loader = DataLoader(dataset, batch_size = batch_size)

    
# 네트워크 생성하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(num_classes=num_classes).to(device)

#손실 함수 정의
criterion = nn.CrossEntropyLoss()
#Optimizer 설정하기
optimizer = optim.Adam(net.parameters(), lr = lr)

step_losses = []
epoch_losses = []

for epoch in tqdm(range(epochs)):
  epoch_loss = 0
  for X,Y in tqdm(data_loader, total=len(data_loader), leave = False):
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()
    Y_pred = net(X)
    loss = criterion(Y_pred, Y)
    print(loss)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    step_losses.append(loss.item())
  epoch_losses.append(epoch_loss/len(data_loader))

working_dir = "working/"

# 모델 파일 경로 설정
model_name = "U-Net.pth"
model_path = os.path.join(working_dir, model_name)
torch.save(net.state_dict(), model_path)


#https://www.kaggle.com/code/dhvananrangrej/image-segmentation-with-unet-pytorch





