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

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    # dataset이 저장될 위치와 transform을 인자로 받음
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # data 파일 리스트 불러오기
        lst_data = os.listdir(self.data_dir)
        # 저장된 이름을 기준으로 불러오기
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)
    # index에 해당하는 파일을 load
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        # 0~1 사이로 normalize
        label = label/255.0
        input = input/255.0
        # neural network의 input은 3개의 axis을 가지고 있어야함
        # 채널이 하나인 경우에도 임의로 생성 해줘야함
        # 차원이 2일 경우 axis를 임의로 생성
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
        # dic 형태로 내보냄
        data = {'input': input, 'label': label}
        # transform이 정의되어 있다면 적용
        if self.transform:
            data = self.transform(data)

        return data
'''
dataset_train = Dataset(data_dir=os.path.join(data_dir), 'train')

data = dataset_train.__getitem__(0)
input = data['input']
label = data['label']

plt.subplot(121)
# 차원제거
plt.imshow(input.squeeze())

plt.subplot(122)
plt.imshow(label.squeeze())

plt.show()
'''

## Transform 구현하기
# data dic to tensor
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        # 순서 x, y, channel -> channel, y, x
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        #numpy to tensor
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        # 50%의 확률로 반전
        if np.random.rand() > 0.5:
            label = np.fliplr(label) #좌우
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label) #위아래
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data