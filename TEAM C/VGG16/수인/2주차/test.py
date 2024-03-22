import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG
from PIL import Image


# Hyperparameters
learning_rate = 0.00005
num_epoch = 50
batch_size = 10

path = './data/'

num_classes = 4
IMG_SIZE = (32, 32)   # resize image

## TRAIN/TEST 데이터셋 정의
# Transform 정의
transform = transforms.Compose( # (데이터 정규화)
    [transforms.Resize(IMG_SIZE),
     transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transforms=None, target_transform=None):
        super(MyDataset, self).__init__()
        with open(root + datatxt, 'r') as f:
            imgs = []
            for line in f:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
            self.root = root
            self.imgs = imgs
            self.transforms = transforms
            self.target_transform = target_transform

    def __getitem__(self, index):
        f, label = self.imgs[index]  
        img = Image.open(self.root + f).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# CIFAR10 TEST 데이터 정의
test_data = MyDataset(path + 'test_data/', 'test_label.txt', transforms=transform)
test_loader = DataLoader(test_data, batch_size=batch_size)

test_loader = DataLoader(test_data, batch_size=batch_size)

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 클래스를 인스턴스화
model = VGG(base_dim=64).to(device)

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 맞은 개수, 전체 개수를 저장할 변수를 지정합니다.
correct = 0
total = 0

# Train
model = VGG(base_dim=64).to(device)
model.load_state_dict(torch.load('./VGG16_newdata.pth'))

# evl
correct = 0
total = 0

model.eval()

# 인퍼런스 모드를 위해 no_grad 해줍니다.
with torch.no_grad():
    # 테스트로더에서 이미지와 정답을 불러옵니다.
    for image,label in test_loader:
        
        # 두 데이터 모두 장치에 올립니다.
        x = image.to(device)
        y= label.to(device)

        # 모델에 데이터를 넣고 결과값을 얻습니다.
        output = model.forward(x)
        _,output_index = torch.max(output,1)

        
        # 전체 개수 += 라벨의 개수
        total += label.size(0)
        correct += (output_index == y).sum().float()
    
    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))