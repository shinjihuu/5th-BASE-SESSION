## 데이터 샘플을 처리하는 코드는 지저분하고 유지 보수가 어려울 수 있기 때문
## 때문에 더 나은 가독성과 모듈성을 위해 데이터셋 코드를 모델 학습 코드로부터 분리하는 것이 이상적

## PyTorch는 torch.utils.data.DataLoader 와 torch.utils.data.Dataset 의 두 가지 데이터 기본 요소를 제공하여 
## 미리 준비해둔(pre-loaded) 데이터셋 뿐만 아니라 가지고 있는 데이터를 사용할 수 있게 함

## Dataset 은 샘플과 정답(label)을 저장하고
## DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌈



## 원래는 클래스로 직접 구현해야 함
'''
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, [내가 필요한 것들]):
        데이터셋 가져와서 선처리
    def __len__(self):
        데이터셋의 길이 적기
    def __getitem__(self,idx): 
        데이터셋에서 한 개의 데이터를 가져오는 함수 정의
'''

## TorchVision 에서 Fashion-MNIST 데이터셋을 불러오는 예제
## 매개변수
   ## root: 학습/테스트 데이터가 저장되는 경로
         ## 가장 중요한 파라미터 -> 이미지 파일의 경로를 알려줌 / class 별로 폴더가 저장되어 있는 경로를 알려주면 됨
   ## train: 학습용 또는 테스트용 데이터셋 여부를 지정 (True면 학습 데이터셋, False면 테스트데이터셋)
   ## download=True: root 에 데이터가 없는 경우 인터넷에서 다운로드
   ## transform, target_transform: 특징(feature)과 정답(label) 변형(transform)을 지정


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

## torchvision의 dataset에 있는 FashionMNIST 데이터셋을 불러옴
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

## 데이터셋에 리스트처럼 접근하고 시각화할 수 있음
## 번호 매핑
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
## 이미지 일부 시각화
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


## DataLoader로 학습용 데이터 준비하기
## DataLoader: 간단한 API로 데이터셋을 불러오는 복잡한 과정들을 추상화한 순회 가능한 객체(iterable)
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
                    ## 한 번에 64개의 이미지를 불러오고 데이터는 무작위로 섞음
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 이미지와 정답(label)을 표시
## DataLoader에서 배치 한 번의 이미지와 레이블을 가져옴
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

## 첫 번째 배치의 첫 번째 이미지 시각화
img = train_features[0].squeeze()
## squeeze()를 사용하면 크기가 1인 차원 (여기서는 채널 수)이 제거됨
## 결과적으로 [높이, 너비] 형태의 2차원 텐서가 됨 -> 이는 이미지를 시각화하기에 적합한 형태
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}") ## 첫 번째 이미지의 레이블 출력