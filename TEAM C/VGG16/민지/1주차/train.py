import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from MyVGG16 import MyVGG

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5]), 
                                transforms.Resize((32,32))]) 
# 각 픽셀의 RGB 픽셀 범위는 0~255임. 이를 각각 Normalize해주는 것
# transforms.Normalize(mean=[0.5], std=[0.5])]) ==> 색상이 표준화됨

# setting
batch_size = 1
lr = 2e-06
epoch = 1

train_dset = datasets.CIFAR10(root='./data/',train=True,transform=transform,target_transform=None,download=True)

train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)


model = MyVGG().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


loss_arr = []
model.train()

for i in tqdm(range(epoch), total=epoch):
    for image, label in train_loader:
        x = image.to(device)
        y = label.to(device)
        optimizer.zero_grad() # optimizer의 gradient를 0으로 설정
        output = model.forward(x)
        loss = loss_func(output, y)
        loss.backward() # 모델의 파라미터에 대한 그래디언트 계산
        optimizer.step() #  그래디언트를 사용하여 모델의 파라미터를 업데이트
    
    #if i % 10 ==0:
    print("loss :", loss)
    # .append(loss.cpu().detach().numpy()) #detach : tensor를 gradient 연산에서 분리
    # 해당 텐서의 연산에 대한 그래디언트 계산을 중단시키기 위함
    # Loss 값을 저장하기 위한 loss_arr에 Loss 값 추가 과정
    # 각 Loss 값은 CPU에 있는 텐서에서 파이썬 넘파이 배열로 변환된 후 저장

torch.save(model, 'VGG_model_01.pth')