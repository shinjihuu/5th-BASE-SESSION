# 데이터로더 & 트랜스폼
import os 
import numpy as np
import torch
import torch.nn as nn

## 데이터 로더 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        lst_data = os.listdir(self.data_dir)
        # 데이터 디렉토리에 저장된 모든 리스트로부터 라벨, 인풋 나누어서 정리
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        # 인덱스 맞게 정렬
        lst_label.sort()
        lst_input.sort()
        # 해당 클래스의 파라미터로 가지고 옴
        self.lst_label = lst_label
        self.lst_input = lst_input

    # 데이터 length 함수
    def __len__(self):
        return len(self.lst_label) 

    # 데이터 get 함수 
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        # 데이터가 0-255 range가지고 있으므로 0-1로 normalize
        label = label/255.0
        input = input/255.0
        # 파이토치에선 뉴럴네트워크에 들어가는 모든 인풋은 3개 axis 가져야 함,
        # x,y,채널
        # 채널 없는경우 채널에 해당하는 axis 무시되는 경우 있으므로 
        # 하나의 채널임에도 불구하고 채널을 임의로 생성해줘야 함
        if label.ndim == 2:
            label = label[:,:, np.newaxis]
        if input.ndim == 2:
            input = input[:,:, np.newaxis]
        data = {'input': input, 'label': label}

        if self.transform: # transform이 있다면
            data = self.transform(data)
            # transform통과한 데이터셋을 다시 받아 리턴
        return data
        

## Transform 구현
# 딕셔너리인 data object를 텐서로 변환
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        # 이미지의 numpy 차원=(y, x, ch), 파이토치 텐서의 차원=(ch, y, x)
        # 순서를 변경합니다.
        #print(label.shape, input.shape)
        label = np.transpose(label, (2, 0, 1)).astype(np.float32)
        input = np.transpose(input, (2, 0, 1)).astype(np.float32)
        
        # 텐서로 변환합니다.
        label = torch.from_numpy(label)
        input = torch.from_numpy(input)

        data = {'label': label, 'input': input}

        return data
    
# normalization
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        input = (input-self.mean) / self.std
        #label은 0,1인 클래스로 정의되어 있으므로 하지않음
        
        data = {'label': label,'input': input}
        
        return data

# 좌우상하로 랜덤 플립
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        # 50%의 확률로 fliplr:좌우로, flipud:위아래로 뒤집음
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)
        
        data = {'label': label,'input': input}
        
        return data