import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 

## 트랜스폼 구현하기
class ToTensor(object): # numpy에서 파이토치 tensor로 변환.
    def __call__(self, data): # input과 label이 있는 데이터를 tensor로 변환.
        label, input = data['label'], data['input'] # 각각의 데이터를 value로 받아서.
        # image dim at np : (Y,X,CH), image dim at pytorch : (CH,Y,X) 이므로...
        label = label.transpose((2, 0, 1)).astype(np.float32) # numpy (x, y, ch) , pytorch (ch, y, x) 
        input = input.transpose((2, 0, 1)).astype(np.float32) # numpy (x, y, ch) , pytorch (ch, y, x)

        data = {'label' : torch.from_numpy(label), 'input' : torch.from_numpy(input)}

        return data


class Normalization(object):
    def __init__(self, mean = 0.5, std = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data): #
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input} # input에만 적용한다 -> label은 0 or 1로 되어있기 때문이다.

        return data

class RandomFlip(object): #  좌우상하 flip
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label) # flip left right
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label) # flip up down
            input = np.flipud(input)    

        data = {'label' : label, 'input' : input}

        return data