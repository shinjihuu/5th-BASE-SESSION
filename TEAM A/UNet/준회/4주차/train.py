## 필요한 패키지 Import
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

from Unet import Unet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data_dir = ''
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)


## K Means Clustering을 통해 이미지가 더 잘 인식될 수 있게 하는 부분 ##
## 임의의 수 지정
num_items = 1000
## 0 ~ 255 사이의 숫자를 3 * num_items 번 랜덤하게 뽑기
color_array = np.random.choice(range(256), 3*num_items).reshape(-1,3)
print(color_array.shape)

num_classes = 10

## K-means Clustering 알고리즘을 사용해 label_model 에 저장
label_model = KMeans(n_clusters = num_classes)
label_model.fit(color_array)


## Dataset 정의하기
class CityscapeDataset(Dataset):

  ## 이미지 디렉토리와 모델이 정의한 라벨 정의하기
  def __init__(self, image_dir, label_model):
    self.image_dir = image_dir
    self.image_fns = os.listdir(image_dir)
    self.label_model = label_model

  ## 이미지 개수 알아보는 함수 정의하기
  def __len__(self):
    return len(self.image_fns)

## 데이터를 이미지와 라벨로 분할하는 함수 정의하기
  def split_image(self, image):
    image = np.array(image)
    cityscape, label = image[:, :256, :], image[:, 256:, :]
    return cityscape, label

## 이미지 텐서로 변환하고 정규화하는 함수 정의하기
  def transform(self, image):
    transform_ops = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform_ops(image)

  ## 인덱스를 통해 이미지를 가져오는 함수 정의하기
  def __getitem__(self, index):
    ## 지정된 인덱스에 해당하는 이미지 파일을 numpy 배열로 변환하는 부분
    image_fn = self.image_fns[index]
    image_fp = os.path.join(self.image_dir, image_fn)
    image = Image.open(image_fp).convert("RGB")
    image = np.array(image)

    ## 데이터를 이미지와 라벨로 분할하는 부분
    cityscape, label = self.split_image(image)

    ## 분할된 라벨 이미지를 모델로 예측 후 long 텐서로 변환하는 부분
    label_class = self.label_model.predict(label.reshape(-1,3)).reshape(256,256)
    label_class = torch.Tensor(label_class).long()
    cityscape = self.transform(cityscape)
    return cityscape, label_class


## 실제로 train 하는 부분 ## 
batch_size = 16
epochs = 10
lr = 0.01

dataset = CityscapeDataset(train_dir, label_model)
data_loader = DataLoader(dataset, batch_size = batch_size)

model = Unet(num_classes = num_classes).to(device)

# 손실함수 정의
criterion = nn.CrossEntropyLoss()
# Optimizer 정의
optimizer = optim.Adam(model.parameters(), lr = lr)

step_losses = []
epoch_losses = []

for epoch in tqdm(range(epochs)):
  epoch_loss = 0
  for X,Y in tqdm(data_loader, total=len(data_loader), leave = False):
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    step_losses.append(loss.item())
  epoch_losses.append(epoch_loss/len(data_loader))


root_dir = "working/"

# 모델 파일 경로 설정
model_name = "Unet.pth"
model_path = os.path.join(root_dir, model_name)
torch.save(net.state_dict(), model_path)


























