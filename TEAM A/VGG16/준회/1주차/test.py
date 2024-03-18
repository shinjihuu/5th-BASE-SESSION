import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 클래스를 인스턴스화
model = VGG(base_dim = 64).to(device)

learning_rate = 0.001

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# CIFAR10 Test 데이터 정의
testset = datasets.CIFAR10(root = "./data", train = False, transform = transform, download = True)
test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 맞은 개수, 전체 개수를 저장할 변수를 지정합니다.

correct = 0
total = 0

model.eval()

# 인퍼런스 모드를 위해 no_grad 해줍니다.
with torch.no_grad():
  # Test_loader에서 이미지와 정답을 불러옵니다.
  for image, label in test_loader:

    # 두 데이터 모두 장치에 올립니다.
    x = image.to(device)
    y = label.to(device)

    # 모델에 데이터를 넣고 결과값을 얻습니다.
    output = model.forward(x)
    _, output_index = torch.max(output, 1)

    # 전체개수 += 라벨의 개수
    total += label.size(0)
    correct += (output_index == y).sum().float()

  # 정확도 도출
  print("Accuary of Test Data: {}%".format(100 * correct / total))

# Inference

import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습용 이미지를 무작위로 가져오기
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))

# 정답(label) 출력
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
