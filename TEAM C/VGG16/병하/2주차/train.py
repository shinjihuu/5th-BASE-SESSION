import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG16
import torch
import torch.nn as nn

# setting
batch_size = 5
learning_rate = 0.001
num_epoch = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') #GPU or CPU 할당 

# Data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

train_data = datasets.ImageFolder(root='data/skintypes/train/',transform=transform) #데이터로드 및 레이블링 기능 제공 class1 - img1,img2 ... 

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

# Train
model = VGG16(base_dim=64,num_classes=3).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []

for i in range(num_epoch): 
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y = label.to(device)

        optimizer.zero_grad() #optimizer 초기화 
        output = model.forward(x)
        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()
        print(f'epoch {i} loss : ',loss)

torch.save(model.state_dict(), "./models/VGG16_newdata.pth") #학습 완료된 모델 저장