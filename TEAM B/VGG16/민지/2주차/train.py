import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from vgg16 import VGG16
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import os
from torch.utils.data import random_split

# setting
batch_size = 16 
learning_rate = 0.0002
num_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

# Data
#__init__ 메서드는 객체를 생성할 때 실행되는 메서드, 즉 생성자입니다. 여기에는 모델에 사용할 데이터를 담아두는 등 어떤 인덱스가 주어졌을 때 반환할 수 있게 만드는 초기 작업을 수행합니다.

#__getitem__ 메서드는 어떤 인덱스가 주어졌을 때 해당되는 데이터를 반환하는 메서드입니다. numpy 배열이나 텐서 형식으로 반환합니다. 보통 입력과 출력을 튜플 형식으로 반환하게 됩니다.

#__len__은 학습에 사용할 데이터의 총 개수라고 볼 수 있는데, 즉 얼마만큼의 인덱스를 사용할지를 반환하는 메서드입니다.

class FlowerPhotosDataset(Dataset):
    def __init__(self, root_dir, transform=None): # 데이터 로드하고 이미지의 경로와 레이블을 저장
        self.root_dir = root_dir #root_dir - 이미지 파일이 있는 디렉토리 경로
        self.transform = transform #transform - 이미지에 적용할 변환
        self.images = [] # 경로 저장할 빈 리스트
        self.labels = [] # 레이블 저장할 빈 리스트 
        self.label_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for label in self.label_names: # 각 클래스 디렉토리에서 이미지 파일을 찾아서 images 리스트에 이미지 파일의 경로를 추가
            label_dir = os.path.join(root_dir, label)
            image_names = [img for img in os.listdir(label_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
            for image_name in image_names:
                self.images.append(os.path.join(label_dir, image_name))
                self.labels.append(self.label_names.index(label))

    def __len__(self): #데이터셋에 포함된 전체 이미지의 수를 반환
        return len(self.images)

    def __getitem__(self, idx): #주어진 인덱스에 해당하는 이미지와 레이블 반환 
        image_path = self.images[idx] # image 리스트에서 해당 인덱스에 있는 이미지 파일의 경로 가져오기 
        image = Image.open(image_path).convert('RGB') # 이미지 파일 열고 rgb형태로 변환하기
        label = self.labels[idx]
        if self.transform: # image에 지정된 transform이 있다면 적용하기 
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)), #이미지는 224 x 224 크기로 조정하고 텐서로 변환
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
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []

for i in range(num_epoch): 
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad() #optimizer의 gradient를 0으로 설정
        output = model.forward(x)
        # _, output = torch.max(output, 1)
        #print(output.shape)
        #print(y_.shape)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()

    if i%10 ==0:
        print(f'epcoh {i} loss : ',loss)
        loss_arr.append(loss.cpu().detach().numpy()) #detach tensor를 gradient 연산에서 분리

torch.save(model.state_dict(), "./VGG16_100")