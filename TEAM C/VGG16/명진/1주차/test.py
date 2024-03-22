import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from VGG16 import VGG16

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cuda gpu사용

#hyperparameter
batch_size = 100

normalize = transforms.Normalize( # 정규화
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)#색상 rgb normalize

# define transforms: 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize, # 정규화
])

# CIFAR10 데이터셋 load
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, 
                                transform=transform, target_transform=None, 
                                download=True)

test_loader = DataLoader(cifar10_test, batch_size=batch_size)


model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load('./train_model/VGG16_01.pth'))

# Test
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for i, (image,label) in enumerate(test_loader):
        images = image.to(device)
        labels = label.to(device)

        output = model.forward(images)
        _, predicted = torch.max(output,1) # 여기서 1은 뭐지? #인덱스 찾는거임

        total += label.size(0)
        correct += (predicted==labels).sum().float()
    # 테스트 데이터셋에 따른 네트워크 정확도(맞춘거/전체)
    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
