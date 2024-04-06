# 라이브러리 추가
import argparse
import os 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

from data_read import * # Dataset & Transform
from tqdm.notebook import tqdm

# Unet 모델 정의
class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.ctr_11 = self.conv_block(in_channels=3, out_channels=64)
        self.ctr_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ctr_21 = self.conv_block(in_channels=64, out_channels=128)
        self.ctr_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ctr_31 = self.conv_block(in_channels=128, out_channels=256)
        self.ctr_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ctr_41 = self.conv_block(in_channels=256, out_channels=512)
        self.ctr_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.exp_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.exp_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.exp_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.exp_22 = self.conv_block(in_channels=512, out_channels=256)
        self.exp_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.exp_32 = self.conv_block(in_channels=256, out_channels=128)
        self.exp_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.exp_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
	# 1x1 convolution layer 추가
        self.output1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    # forward pass
    def forward(self, X):
        ctr_11_out = self.ctr_11(X) # [-1, 64, 256, 256]
        ctr_12_out = self.ctr_12(ctr_11_out) # [-1, 64, 128, 128]
        ctr_21_out = self.ctr_21(ctr_12_out) # [-1, 128, 128, 128]
        ctr_22_out = self.ctr_22(ctr_21_out) # [-1, 128, 64, 64]
        ctr_31_out = self.ctr_31(ctr_22_out) # [-1, 256, 64, 64]
        ctr_32_out = self.ctr_32(ctr_31_out) # [-1, 256, 32, 32]
        ctr_41_out = self.ctr_41(ctr_32_out) # [-1, 512, 32, 32]
        ctr_42_out = self.ctr_42(ctr_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(ctr_42_out) # [-1, 1024, 16, 16]
        exp_11_out = self.exp_11(middle_out) # [-1, 512, 32, 32]
        exp_12_out = self.exp_12(torch.cat((exp_11_out, ctr_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        exp_21_out = self.exp_21(exp_12_out) # [-1, 256, 64, 64]
        exp_22_out = self.exp_22(torch.cat((exp_21_out, ctr_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        exp_31_out = self.exp_31(exp_22_out) # [-1, 128, 128, 128]
        exp_32_out = self.exp_32(torch.cat((exp_31_out, ctr_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        exp_41_out = self.exp_41(exp_32_out) # [-1, 64, 256, 256]
        exp_42_out = self.exp_42(torch.cat((exp_41_out, ctr_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(exp_42_out) # [-1, 64, 256, 256] -> [-1, 64, 256, 256]
        output_out1 = self.output(output_out) # [-1, num_classes, 256, 256]
        
        return output_out1
    
# GPU 사용
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

#하이퍼파라미터
batch_size = 4

epochs = 10
lr = 0.01
#데이터로더
dataset = CityscapeDataset(train_dir, label_model)
data_loader = DataLoader(dataset, batch_size = batch_size)

model = UNet(num_classes = num_classes).to(device)

# 손실함수 정의
criterion = nn.CrossEntropyLoss()
# Optimizer 정의
optimizer = optim.Adam(model.parameters(), lr = lr)

#### TRAIN
step_losses = []
epoch_losses = []

for epoch in tqdm(range(epochs)) :
  epoch_loss = 0
  
  for X, Y in tqdm(data_loader, total = len(data_loader), leave = False) :
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    step_losses.append(loss.item())
  epoch_losses.append(epoch_loss/len(data_loader))

print(len(epoch_losses))
print(epoch_losses)
# 모델 저장
root_path = data_dir
model_name = "UNet.pth"
torch.save(model.state_dict(), root_path + model_name)


#### VAL
#저장된 모델 불러오기
model_path = root_path + model_name
model_ = UNet(num_classes = num_classes).to(device)
model_.load_state_dict(torch.load(model_path))

# val 데이터셋
test_batch_size = 8
dataset = CityscapeDataset(val_dir, label_model)
data_loader = DataLoader(dataset, batch_size = test_batch_size)

X,Y = next(iter(data_loader))
X,Y = X.to(device), Y.to(device)
Y_pred = model_(X)
print(Y_pred.shape)
Y_pred = torch.argmax(Y_pred, dim=1)
print(Y_pred.shape)

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])
fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))
# IoU계산
iou_scores = []

for i in range(test_batch_size):
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    
    # IOU score
    intersection = np.logical_and(label_class, label_class_predicted)
    union = np.logical_or(label_class, label_class_predicted)
    iou_score = np.sum(intersection) / np.sum(union)
    iou_scores.append(iou_score)

    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")

plt.show()

