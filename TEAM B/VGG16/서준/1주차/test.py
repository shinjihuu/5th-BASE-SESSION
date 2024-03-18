import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg16 import VGG16
import torch
import torch.nn as nn

# device 설정
# 만약 cuda 설정이 되어있지 않다면 cpu를 사용한다
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

# 배치 사이즈 설정
batch_size = 100

# 방대한 데이터 이미지를 한번에 변형 (데이터 전처리)
transform = transforms.Compose(
    [transforms.ToTensor(), # 이미지를 Pytorch tensors 타입으로 변형 / pixels 값들을 [0~255]에서 [0.0~1.0]으로 자동 변환
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # 각 channel에 맞춰서 normalize 진행
)

# cifar10 데이터세트를 Data폴더 안에 불러오기
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

# cifar10_test 데이터를 미리 설정한 batch_size(100) 형태로 만들어줌
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# VGG16 모델 불러오기
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load('./train_model/VGG16_100.pth'))

# 모델 평가에 사용될 변수 선언 및 모델 평가하기
correct = 0
total = 0
model.eval()

# torch.no_grad() : 기울기(gradient)를 계산하지 않는다.
with torch.no_grad():
    for i, [image, label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x) # forward() : 모델이 학습 데이터를 입력 받아서 forward 연산 수행 / forward를 호출하여 입력 데이터를 모델에 전달
        _, output_index = torch.max(output, 1)  # 모델의 출력을 기반으로 예측 수행. torch.max 함수를 사용하여 출력 텐서 output에서 가장 큰 값의 인덱스를 찾는다.
                                                # torch.max(output, 1) : 각 행에서 (최대값, 해당 인덱스) 반환
        # 모델 평가에서 선언한 변수들의 값 계산 ()
        total += label.size(0)
        correct += (output_index == y).sum().float()

    # 모델 정확도 출력
    print("Accuracy of Test Data: {}%".format(100 * correct / total))