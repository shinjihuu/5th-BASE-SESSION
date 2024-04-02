from dataset import val_dataset
from unet import UNet
from torch.utils.data import DataLoader
from train import criterion, optimizer
from torchvision import transforms
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

num_classes = 30
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = os.path.join('./ckpt','unetepoch20.pth')
model_ = UNet(n_channels=3,n_classes=num_classes,bilinear=True).to(device)
model_.load_state_dict(torch.load(model_path))

data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def inference():

    X, Y = next(iter(data_loader))
    X, Y = X.to(device), Y.to(device)
    Y_pred = model_(X)
    #print(Y_pred.shape)
    Y_pred = torch.argmax(Y_pred, dim=1)
    #print(Y_pred.shape)

    inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])
    fig, axes = plt.subplots(batch_size, 3, figsize=(3*5, 8*5))

    for i in range(8):
        
        landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
        label_class = Y[i].cpu().detach().numpy()
        label_class_predicted = Y_pred[i].cpu().detach().numpy()
        
        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title("Landscape")
        axes[i, 1].imshow(label_class)
        axes[i, 1].set_title("Label Class")
        axes[i, 2].imshow(label_class_predicted)
        axes[i, 2].set_title("Label Class - Predicted")

    fig.savefig("inference.png")

if __name__ == '__main__':
    inference()