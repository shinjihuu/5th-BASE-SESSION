import os
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset

class CustomMNIST(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()  # 부모 클래스의 생성자 호출
        self.MNIST_train = MNIST(root=root,
                                 train=train,
                                 download=False if os.path.exists(os.path.join(root, './data/MNIST/raw/')) else True,
                                 transform=transform)
        self.MNIST_test = MNIST(root=root, 
                                train=False, 
                                download=False if os.path.exists(os.path.join(root, './data/MNIST/raw/')) else True,
                                transform=None)
        self.transform = transform if transform is not None else self.preprocess() 
        # 디폴트가 None이므로, preprocess() 함수 호출하여 transform 값 가져옴.

    def preprocess(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 데이터셋의 평균과 표준편차로 정규화
        ])
        return transform

    def __len__(self):
        return len(self.MNIST_train)

    def __getitem__(self, idx):
        image, label = self.MNIST_train[idx]
        if self.transform:
            image = self.transform(image)
        return image, label