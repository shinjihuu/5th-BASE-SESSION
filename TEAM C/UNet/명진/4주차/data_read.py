# 데이터 전처리
# 필요한 패키지들을 불러옵니다.
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

# 데이터 디렉토리 설정
data_dir = './cityscapes_data'
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# 훈련 및 검증 데이터 파일 로드
# train_dir/ val_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 train_fns에 저장
train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)
print(len(train_fns), len(val_fns))

# 데이터 수와 색상 배열 설정
num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

# 클래스 수 설정
num_classes = 10
#  K-means 클러스터링
label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)

# 도시스케이프 데이터셋 클래스 정의
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
        image = Image.open(image_fp)
        image = np.array(image)
        cityscape, label = self.split_image(image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        label_class = torch.Tensor(label_class).long()
        cityscape = self.transform(cityscape)
        return cityscape, label_class

    def split_image(self, image):
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

# 데이터셋 생성
dataset = CityscapeDataset(train_dir, label_model)
print(len(dataset))

# 샘플 데이터 확인
cityscape, label_class = dataset[0]
print(cityscape.shape)
print(label_class.shape)