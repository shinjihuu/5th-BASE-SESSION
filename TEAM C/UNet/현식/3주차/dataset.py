import os
import numpy as np

import torch
import torch.nn as nn

# 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) #데이터 디렉토리에서 파일 목록을 가져옴

        #파일 목록 중 'label'으로 시작하는 파일만 선택
        lst_label = [f for f in lst_data if f.startswith('label')]
        #파일 목록 중 'input'으로 시작하는 파일만 선택
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index): #실제로 데이터를 get하는 함수
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0 #0~1사이로 정규화(픽셀 값이 0부터 255사이의 정수로 표현)
        input = input/255.0 #0~1사이로 정규화(픽셀 값이 0부터 255사이의 정수로 표현)

        if label.ndim == 2:
            label = label[:, :, np.newaxis] #label 데이터가 2차원인 경우 새로운 축을 추가하여 데이터를 3차원으로 만듦(모든 이미지 데이터가 동일한 차원을 갖도록 함)
        if input.ndim == 2:
            input = input[:, :, np.newaxis] #input 데이터가 2차원인 경우 새로운 축을 추가하여 데이터를 3차원으로 만듦(모든 이미지 데이터가 동일한 차원을 갖도록 함)

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## transform 구현하기
class ToTensor(object): #numpy->tensor
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) #tesnor 형식(채널, Y, X)
        input = input.transpose((2, 0, 1)).astype(np.float32) #tensor 형식(채널, Y, X)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object): #정규화
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std #label은 0또는 1값이므로 정규화 X

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object): #데이터 증강을 위해 이미지를 무작위로 수평 또는 수직으로 뒤집음
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data