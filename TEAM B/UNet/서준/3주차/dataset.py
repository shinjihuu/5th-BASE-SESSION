import os
import numpy as np

import torch
import torch.nn as nn

## Data Loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # self.data_dir 속에 있는 파일들 리스트로 받아오기
        lst_data = os.listdir(self.data_dir)

        # startswith : 현재 문자열이 사용자가 지정하는 특정 문자로 시작하는지 확인 / 리턴값 : True or False
        # startswith 값을 판단하여 반복
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        # 파일명 정렬
        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

        def __len__(self):
            # lst_label의 길이
            return len(self.lst_label)
        
        def __getitem__(self, index):
            # data_dir에서 인덱스에 해당하는 레이블과 입력 데이터를 numpy 배열로 불러옴
            label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
            input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

            # label과 입력 데이터 정규화
            label = label/255.0
            input = input/255.0

            # ndim : 어떤 array가 몇차원인지를 반환
            # 레이블과 입력 데이터의 차원 확인 및 조정
            if label.ndim == 2:
                # np.newaxis : 존재하는 numpy array의 차원을 늘려줌 (1D -> 2D / 2D -> 3D / 3D -> 4D)
                label = label[:, :, np.newaxis]
            if input.ndim == 2:
                input = input[:, :, np.newaxis]

            # 정보를 담아 딕셔너리 생성
            data = {'input': input, 'label': label}

            # 변환이 정의 되어있다면 데이터 변환 수행
            if self.transform:
                data = self.transform(data)

            return data
        
## 트렌스폼 구현
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 차원 변경 및 데이터 타입 변환 (float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # Tensor로 변환
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
    
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 입력 데이터 정규화
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data
    
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 랜덤으로 좌우 or 상하 반전
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliprl(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data