
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 

from unet import UNet
from train import CityscapeDataset
from train import label_model, num_classes

data_dir = os.path.join("datasets/cityscapes_data")
train_dir = os.path.join(data_dir, "train") 
val_dir = os.path.join(data_dir, "val")

train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "working/U-Net.pth"
model_ = UNet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_path))

test_batch_size = 10
dataset = CityscapeDataset(val_dir, label_model)
data_loader = DataLoader(dataset, batch_size=test_batch_size)


X,Y = next(iter(data_loader))
X,Y = X.to(device), Y.to(device)
Y_pred = model_(X)
#각 픽셀에 대한 가장 높은 확률을 가진 클래스의 인덱스를 반환
Y_pred = torch.argmax(Y_pred, dim=1)
print(Y_pred.shape)

#역변환은 주어진 이미지에 적용된 정규화(normalization)를 반대로 적용하여 원래의 이미지 픽셀 값 범위로 되돌리는 것을 의미
inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])


fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))

iou_scores = []

for i in range(test_batch_size):
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    #원래 이미지 픽셀 값 범위로 변환
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    
    # IOU score
    intersection = np.logical_and(label_class, label_class_predicted)
    union = np.logical_or(label_class, label_class_predicted)
    iou_score = np.sum(intersection) / np.sum(union)
    iou_scores.append(iou_score)


    print(axes[i, 0].imshow(landscape))
    print(axes[i, 0].set_title("Landscape"))
    print(axes[i, 1].imshow(label_class))
    print(axes[i, 1].set_title("Label Class"))
    print(axes[i, 2].imshow(label_class_predicted))
    print(axes[i, 2].set_title("Label Class - Predicted"))
    
print(sum(iou_scores) / len(iou_scores))
