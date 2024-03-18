import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from Vgg import VGG16

import argparse

# device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = VGG16(init_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test(model, test_loader):
    correct, total = 0, 0

    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y = label.to(device)

            output = model.forward(x)
            _, output_idx = torch.max(output,1)

            total += label.size(0)
            correct += (output_idx == y).sum().float()
        print(f'Accuracy of Test Data :{100*correct/total}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VGG16 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (.pth)")
    
    args = parser.parse_args()
    
    # 모델 로드
    model = load_model(args.model_path)
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, target_transform=None, download=True)
    test_loader = DataLoader(cifar10_test, batch_size=128)
    
    # 테스트 실행
    test(model, test_loader)