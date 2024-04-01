import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from custom_dataset import CustomImageDataset # 데이터셋 클래스 사용

from vgg16 import VGG16  # VGG16 모델 정의를 포함한 파일

# GPU 사용 가능 여부에 따라 device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 학습 변수 설정
num_epoch = 10
learning_rate = 0.0002
batch_size = 32

# MNIST dataset은 28X28, 1채널(흑백)
# 이미지 전처리 단계 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)), # 이미지 크기 224x224로 조정
    transforms.Grayscale(num_output_channels=3), # 흑백 이미지를 3채널로 변환
    transforms.ToTensor(), # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.5,), (0.5,)) # 흑백 이미지이므로 채널 하나만 정규화
])

# MNIST 데이터셋 로딩
mnist_train = datasets.MNIST(root='./Data/', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./Data/', train=False, transform=transform, download=True)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size)

# VGG16 모델 인스턴스 생성 및 device에 할당
model = VGG16(base_dim=64, num_classes=10).to(device) # MNIST는 10개 클래스를 가지므로 num_classes=10

# 손실 함수와 최적화 알고리즘 설정
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 과정
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
    if epoch % 2 == 0:
        print(f'Epoch [{epoch}/{num_epoch}], Loss: {loss.item()}')

# 학습 완료 후 모델의 가중치 저장
torch.save(model.state_dict(), "./VGG16_MNIST.pth")
