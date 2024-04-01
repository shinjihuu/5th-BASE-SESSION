import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from custom_dataset import CustomImageDataset # 데이터셋 클래스 사용

from vgg16 import VGG16  # VGG16 모델 정의를 포함한 파일

# GPU 사용 가능 여부에 따라 device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 테스트 데이터에 적용할 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기를 224x224로 조정
    transforms.Grayscale(num_output_channels=3),  # 흑백 이미지를 3채널로 변환
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 흑백 이미지이므로 채널 하나만 정규화
])

# 테스트 데이터셋 로딩
mnist_test = datasets.MNIST(root="./Data/", train=False, transform=transform, download=True)

test_loader = DataLoader(mnist_test, batch_size=32)

# 모델 인스턴스 생성 및 저장된 가중치 로드
model = VGG16(base_dim=64, num_classes=10).to(device)
model.load_state_dict(torch.load("./VGG16_MNIST.pth"))
model.eval()  # 모델을 평가 모드로 설정

# 테스트 데이터셋에 대한 모델의 정확도 계산
correct = 0
total = 0

with torch.no_grad():  # 그래디언트 계산을 비활성화
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the MNIST test images: {accuracy:.2f}%')