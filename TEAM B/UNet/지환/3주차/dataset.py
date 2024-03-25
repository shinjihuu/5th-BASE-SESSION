### X:AI 3주차 Code 과제
### AI빅데이터융합경영 배지환 

import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) # data_dir에 있는 파일 이름들을 list로 반환

        lst_label = [f for f in lst_data if f.startswith('label')] # label로 시작하는 파일 이름들 list로 반환
        lst_input = [f for f in lst_data if f.startswith('input')] # input으로 시작하는 파일 이름들 list로 반환

        lst_label.sort() # 정렬
        lst_input.sort()

        self.lst_label = lst_label # class 파라미터 선언 
        self.lst_input = lst_input

    def __len__(self): # len 함수 구현
        return len(self.lst_label)

    def __getitem__(self, index): # index에 해당하는 file return 함수 구현
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 0~1 Normalize
        label = label/255.0 # image data가 0~255 범위로 표현됨
        input = input/255.0

        if label.ndim == 2: # data가 2차원 배열(gray)이면 새로운 axis 추가
            label = label[:, :, np.newaxis] # np.newaxis 새로운 axis 생성
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label} # 생성한 label, input dict 생성

        if self.transform: # transform 설정되어 있으면 transform 적용
            data = self.transform(data)

        return data


## trasform 구현하기
    
# numpy 배열을 tensor로 변환
class ToTensor(object): 
    def __call__(self, data): # data dict
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) # (높이, 너비, 채널) = (0, 1, 2) 순서를 (채널, 높이, 너비) = (2, 0, 1) 순서로 변환
        input = input.transpose((2, 0, 1)).astype(np.float32) # PyTorch에서는 이미지 데이터를 (채널, 높이, 너비) 순서로 다룸

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)} # torch.from_numpy : numpy 배열을 tensor로 변환하는 함수

        return data

# 입력 이미지 정규화
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

# 입력 이미지 random 반전
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5: # 0.5보다 크면 좌우반전
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5: # 0.5보다 크면 상하반전
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data