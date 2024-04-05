# 필요한 라이브러리
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) # listdir : 데이타 디렉토리의 모든 파일들의 리스트를 불러움

        lst_label = [f for f in lst_data if f.startswith('label')] # pre fixed 되어있는 데이터만 부름
        lst_input = [f for f in lst_data if f.startswith('input')] # startswith가 그 역할임.

        lst_label.sort() # 인덱스에 맞게 정렬
        lst_input.sort()

        self.lst_label = lst_label # 정렬된 리스트를 파라미터로 지정하도록 함.
        self.lst_input = lst_input # 
    
    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index): # 실제로 데이터를 get하는 함수
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])) # numpy형태로 데이터가 저장되어 있기 때문에 np.load
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0 # 0~1로 정규화
        input = input/255.0

        # NN에 들어가는 모든 input은 3차원이어야 함.
        if label.ndim == 2: # 채널이 없는 경우, 채널에 해당하는 축을 임의로 생성해야함.
            label = label[:,:,np.newaxis] # 임의로 없던 축을 만들어준다~
        if input.ndim == 2: 
            input = input[:,:,np.newaxis] # input도 마찬가지 없던 축 생성

        data = {'input': input, 'label': label} # 딕셔너리 형태로 내보냄.

        if self.transform: # 
            data = self.transform(data) # 트랜스폼 함수가 존재한다면, 트랜스폼한 결과를 리턴해라.
            
        return data