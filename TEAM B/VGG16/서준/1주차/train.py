import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg16 import VGG16
import torch
import torch.nn as nn

batch_size = 100 # 배치 사이즈 설정

# learning rate : 모델이 학습을 진행할 때 각각의 가중치(weight)를 얼마나 업데이트할지 결정하는 하이퍼파라미터
learning_rate = 0.0002 # learning rate 설정

num_epoch = 100 # 학습 횟수 설정

# device 설정
# 만약 cuda 설정이 되어있지 않다면 cpu를 사용한다
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

# 방대한 데이터 이미지를 한번에 변형 (데이터 전처리)
transform = transforms.Compose(
    [transforms.ToTensor(), # 이미지를 Pytorch tensors 타입으로 변형 / pixels 값들을 [0~255]에서 [0.0~1.0]으로 자동 변환
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # 각 channel에 맞춰서 normalize 진행
)

cifar10_train = datasets.CIFAR10(root='./Data/',train=True,transform=transform,target_transform=None,download=True)
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

# cifar10_train, cifar10_test 데이터를 미리 설정한 batch_size(100) 형태로 만들어 train_loader, test_loader 변수에 넣는다
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# VGG16 모델 불러오기
model = VGG16(base_dim=64).to(device)

# 손실 함수(loss_function)을 이진분류 함수로 설정
loss_func = nn.CrossEntropyLoss()

# 최적화 알고리즘(optimizer)을 Adam으로 설정
# learning rate : 모델이 학습을 진행할 때 각각의 가중치(weight)를 얼마나 업데이트할지 결정하는 하이퍼파라미터
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []

# num_epoch 만큼 학습 진행
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)

        # torch.no_grad() : 기울기(gradient)를 계산하지 않는다.
        optimizer.zero_grad() #optimizer의 gradient를 0으로 설정

        output = model.forward(x)  # forward() : 모델이 학습 데이터를 입력 받아서 forward 연산 수행

        # 손실 값(loss) 저장
        loss = loss_func(output,y_)

        # backward() : 스칼라 값에 대한 출력 텐서의 gradient를 전달 받고, 동일한 스칼라 값에 대한 입력 텐서의 변화도 계산
        loss.backward()

        # 역전파 단계에서 수집된 변화도( backward() )로 매개변수 조정
        optimizer.step()

    # 학습 횟수가 10의 배수가 될때마다 출력
    if i % 10 ==0:
        print(f'epcoh {i} loss : ',loss) # 손실 값 출력


        # GPU 메모리에 올려져 있는 tensor를 numpy로 변환
        # cpu().detach().numpy()보단 detach().cpu().numpy()를 주로 사용하는 것 같다.
        loss_arr.append(loss.cpu().detach().numpy())

# 모델 저장
torch.save(model.state_dict(), "./train_model/VGG16_100.pth")