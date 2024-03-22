import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset 
from vgg16 import VGG16
import numpy as np
import torch 
import torch.nn as nn 

import os #파일 경로를 다루는 모듈
from PIL import Image #이미지 처리하기 위한 라이브러리

#SVHN 데이터는 '.mat'파일로 제공되어 해당 파일 형태를 읽기 위한 모듈
#.mat : MATLAB에서 사용하는 파일 형식
#MATLAB: 이미지 처리,컴퓨터 비전 등 다양한 분야에서 사용되는 고성능 수치 컴퓨팅 환경 및 프로그래밍 언어
import scipy.io as sio
 
#setting
batch_size = 100 
learning_rate = 0.0002 
num_epoch = 100 

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

#Dataset Class(.mat)
class MyDataset(Dataset): #Dataset 클래스 상속
    def __init__(self,root_dir,data_key,label_key,split='train',transform=None): 
        '''
        root_dir: Dataset 파일이 위치한 디렉토리 경로
        data_key: .mat 파일 내에서 이미지 데이터를 참조하는 키
        label_key: .mat 파일 내에서 레이블 데이터를 참조하는 키
        split: 'train' 또는 'test' 등 데이터셋의 구분
        transform: 샘플에 적용될 선택적 변환
        '''
        self.root_dir = root_dir
        self.data_key = data_key
        self.label_key = label_key
        self.split = split
        self.transform = transform
        #초기화 시, 'load_data' 메서드를 호출하여 데이터셋 load
        #이미지 데이터와 레이블을 self.data와 self.labels에 저장
        self.data, self.labels = self.load_data()  

    def load_data(self):
        data_file = os.path.join(self.root_dir, f'{self.split}_32x32.mat')
        mat = sio.loadmat(data_file) #scipy의 'loadmat'함수를 사용하여 '.mat'파일 load
        
        images = mat[self.data_key]
        labels = mat[self.label_key].astype('int').squeeze()
        labels[labels == 10] = 0 #레이블에서 10을 0으로 매핑하여, 클래스 레이블이 0부터 시작하도록 조정
    
        # 이미지 데이터의 차원을 (높이, 너비, 채널)에서 (배치 크기, 채널, 높이, 너비)로 변환
        images = np.transpose(images, (3, 2, 0, 1)) if images.ndim == 4 else images
        return images, labels

    def __len__(self): #데이터셋의 총 샘플 수 반환
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx] #특정 인덱스 idx에 해당하는 샘플(이미지와 레이블)을 데이터셋에서 가져옴
        label = self.labels[idx]
        
        # 이미지가 PIL.Image가 아닐 경우 변환
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.transpose((1, 2, 0)))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

transforms = transforms.Compose( 
    [transforms.Resize((32,32)),
     transforms.ToTensor(), 
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] 
)

#SVHN(Street View Hous Numbers) 데이터셋 load
#구글이 구글 지도를 만드는 과정에서 촬영한 영상에서 집들의 번호판을 찍어 놓은 32x32 크기의 RGB 데이터
svhn_train = MyDataset(root_dir='./Data/', data_key='X', label_key='y',split='train', transform=transforms)
svhn_test = MyDataset(root_dir='./Data/', data_key='X', label_key='y',split='test', transform=transforms)

train_loader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(svhn_test, batch_size=batch_size)

#Train
model = VGG16(base_dim=64).to(device) 
loss_func = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

loss_arr = [] 

for i in range(num_epoch): 
    for j,[image,label] in enumerate(train_loader): 
        x = image.to(device)
        y = label.to(device) 

        optimizer.zero_grad() 
        output = model.forward(x) 
        loss = loss_func(output,y) 
        loss.backward() 
        optimizer.step() 

    if i%10 == 0 : 
        print(f'epoch {i} loss : ', loss) 
        loss_arr.append(loss.cpu().detach().numpy()) 

#모델의 학습된 가중치들을 저장
torch.save(model.state_dict(), "./svhn_train")