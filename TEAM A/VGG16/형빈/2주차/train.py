from vgg16 import vgg

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from bird_data import train_data_loader, valid_data_loader



learning_rate = 0.001
num_epoch = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vgg(dim=64).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)



epochs = 20


def train():
    train_loss = []
    val_loss = []

    for i in range(epochs):
        _iter = 1
        start_time = time.time()
        train_epoch_loss = []
        val_epoch_loss = []
        start = time.time()
        for image, label in train_data_loader:
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)

            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            train_epoch_loss.append(loss_value)

            if _iter % 500 == 0:
                print("> Iteration {} < ".format(_iter))
                print("Iter Loss = {}".format(round(loss_value, 4)))
        
            _iter += 1

        for image, label in valid_data_loader:

            images = image.to(device)
            labels = label.to(device)


            preds = model.forward(images)


            loss = loss_func(preds,labels)

            loss_value = loss.item()
            val_epoch_loss.append(loss_value)
       

        train_epoch_loss = np.mean(train_epoch_loss)

        val_epoch_loss = np.mean(val_epoch_loss)
        end = time.time()

        train_loss.append(train_epoch_loss)
    
        val_loss.append(val_epoch_loss)
    
        print("** Epoch {} ** - Epoch Time {}".format(i, int(end-start)))
        print("Train Loss = {}".format(round(train_epoch_loss, 4)))
        print("Val Loss = {}".format(round(val_epoch_loss, 4)))

    return model



if __name__ =='__main__' :
    model = train()

torch.save(model.state_dict(), "./trained_model/vgg16_bird.pth")


# https://www.kaggle.com/code/jainamshah17/pytorch-starter-image-classification