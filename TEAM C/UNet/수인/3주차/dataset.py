import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

## 데이터 로더 구현하기 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) # 해당 디렉토리의 파일 리스트

        # prefixed 되어 있는 리스트들 저장
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        # 정렬한 리스트들을 해당 클래스의 파라미터로 가지고 있기
        self.lst_label = lst_label
        self.lst_input = lst_input
    
    def __len__(self): 
        return len(self.lst_label)
    
    def __getitem__(self, index): # 실제로 데이터를 get하는 함수
        label = np.load(os.path.join(self.data_dir, self.lst_label(index)))
        input = np.load(os.path.join(self.data_dir, self.lst_input(index)))

        # normalize
        label = label/255.0
        input = input/255.0

        # 채널에 관한 ? 자동적으로 없던 axis가 생기게 함
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label} # 딕셔너리 형태로 저장

        # transform 함수가 정의되어 있다면, 이 함수를 통과한 데이터 리턴하기
        if self.transform:
            data = self.transform(data)

        return data

## 세 가지의 transform 구현하기 
class ToTensor(object): # ToTensor(): numpy -> tensor
    def __call__(self, data):
        label, input = data['label'], data['input']

        # Image의 numpy 차원 = (Y, X, CH)
        # Image의 tensor 차원 = (CH, Y, X)
        # CH 위치 옮기기
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
    
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std # label은 0 또는 1이라서 안함

        data = {'label': label, 'input': input}

        return data
    
class RandomFlip(object): # RandomFlip()
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5: # 50%의 확률로 
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data
    
## Test 
# Compose: 여러 Transform 함수들을 묶어서 사용할 수 있는 함수 
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

## Dataset 클래스에 해당하는 object 만들기
dataset_train = Dataset(data_dir = os.path.join(data_dir, 'train'), transform = transform)

## variable을 하나 가져오기
data = dataset_train.__getitem__(0)

label = data['label']
input = data['input']

## 시각화
plt.subplot(121)
plt.imshow(input.sqeeze()) # sqeeze(): 배열에서 크기가 1인 차원을 제거하는 함수

plt.subplot(122)
plt.imshow(label.sqeeze())

plt.show()

