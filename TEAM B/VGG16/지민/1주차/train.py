import torch
import torch.nn as nn
import torchvision.transforms as transforms

from vgg16 import VGG16
from torch.utils.data import DataLoader
from torchvision import datasets

# GPU 사용 가능 여부에 따라 device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 학습 변수 설정
num_epoch = 10 # 전체 데이터셋을 몇 번 반복해서 학습할
learning_rate = 0.0002 # 학습률 - 학습 과정에서 사용할 스텝의 크기 / 크게 하면 학습 어려움
batch_size = 32 # 한 번에 학습할 데이터의 수

# 이미지 전처리 단계 정의
transform = transforms.Compose([
    transforms.ToTensor(), # 이미지를 PyTorch 텐서로 변환
    transforms.Resize((224, 224)), # 이미지 크기 224x224로 조정
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 이미지 픽셀 값 범위 정규화 -> 픽셀 값의 범위를 [0, 1]에서 [-1, 1]로 정규화??
])

# 훈련 데이터셋 로딩
train_data = datasets.ImageFolder('./Data/cat_dog/training_set', transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# VGG16 모델 인스턴스 생성 및 device에 할당
model = VGG16(base_dim=64).to(device)
# 손실 함수로 CrossEntropyLoss를 사용 -> 다중 클래스 분류 문제에 적합
loss_func = nn.CrossEntropyLoss()
# 최적화 알고리즘으로 Adam을 사용
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 과정
for i in range(num_epoch): # 에폭만큼 반복하여 모델 훈련
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device) # 이미지 데이터를 현재 device로 이동
        y = label.to(device) # 레이블 데이터를 현재 device로 이동

        optimizer.zero_grad() # 이전 그래디언트를 초기화 -> 이전 반복에서의 그래디언트가 다음 반복에 영향을 주지 않도록

        output = model(x) # 모델에 이미지 데이터를 입력하여 예측값 산출
        loss = loss_func(output, y) # 실제 레이블과 예측 값을 비교하여 손실 계산

        loss.backward() # 손실 함수의 그래디언트를 계산 (역전파)
        optimizer.step() # 모델의 가중치 업데이트
    
    if i % 2 == 0: # 2 에폭마다 손실값 출력 / 10에폭마다?
        print(f'epoch {i} loss: {loss.item()}')

# 학습 완료 후 모델의 가중치 저장   
torch.save(model.state_dict(), "./VGG16_100.pth")  