import os
import numpy as np
import torch
import torch.nn as nn

# 사용자 정의 데이터셋 클래스
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir  # 데이터가 위치한 디렉토리 경로
        self.transform = transform  # 전처리 transform 목록

        lst_data = os.listdir(self.data_dir)  # 데이터 디렉토리 내 모든 파일 리스트

        lst_label = [f for f in lst_data if f.startswith('label')]  # 라벨 데이터 파일
        lst_input = [f for f in lst_data if f.startswith('input')]  # 입력 데이터 파일

        lst_label.sort()  # 파일 이름으로 정렬
        lst_input.sort()  

        self.lst_label = lst_label  # 정렬된 라벨 파일 리스트
        self.lst_input = lst_input  # 정렬된 입력 파일 리스트

    def __len__(self):
        # 데이터셋의 길이는 라벨 리스트의 길이와 같음
        return len(self.lst_label)

    def __getitem__(self, index):
        # 인덱스에 해당하는 라벨과 입력 데이터 로드
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 정규화 -> 데이터 값을 0에서 1 사이로 스케일링
        label = label / 255.0
        input = input / 255.0

        # 2D 데이터의 경우 마지막에 차원 추가
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        # 전처리 변환 적용
        if self.transform:
            data = self.transform(data)

        return data

# torch tensor로 변환하는 전처리 클래스
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 데이터 차원 변환 (채널, 높이, 너비)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

# 정규화를 수행하는 전처리 클래스
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean  
        self.std = std    

    def __call__(self, data):
        label, input = data['label'], data['input']

        # 입력 데이터만 정규화
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

# 무작위로 데이터를 뒤집는 전처리 클래스
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 50% 확률로 좌우 반전
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        # 50% 확률로 상하 반전
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data
