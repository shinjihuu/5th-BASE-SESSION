import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg16 import VGG16
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import os
from torch.utils.data import random_split

#setting
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 100

# Data
class FlowerPhotosDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for label in self.label_names:
            label_dir = os.path.join(root_dir, label)
            image_names = [img for img in os.listdir(label_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
            for image_name in image_names:
                self.images.append(os.path.join(label_dir, image_name))
                self.labels.append(self.label_names.index(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

flower_dataset = FlowerPhotosDataset(root_dir="/home/work/XAI/XAI/week2/paper/code/flower_photos", transform=transform)

# 전체 데이터셋 사이즈 정의
dataset_size = len(flower_dataset)

# 학습 및 테스트 세트 비율 정의
train_size = int(dataset_size * 0.8)  # 예: 전체 데이터셋의 80%를 학습용으로 사용
test_size = dataset_size - train_size  # 나머지 20%를 테스트용으로 사용

# 데이터셋을 무작위로 분할
flower_train, flower_test = random_split(flower_dataset, [train_size, test_size])

# 각 세트에 대한 DataLoader 생성
train_loader = DataLoader(flower_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(flower_test, batch_size=batch_size)

# Train
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load(('./VGG16_100')))

# eval
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for i, [image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        _, output_index = torch.max(output,1) 

        total += label.size(0)
        correct += (output_index==y).sum().float()
    
    print("Accuracy of Test Data: {}%".format(100*correct/total))