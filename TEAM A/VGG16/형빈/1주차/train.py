from vgg16 import vgg

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from data import train_loader



learning_rate = 0.001
num_epoch = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vgg(dim=64).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)





def train():


    start_time = time.time()

    loss_arr = []
    for i in range(num_epoch):
        for j, [image, label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())

    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total execution time: {total_time} seconds")

    return model



if __name__ =='__main__' :
    model = train()

torch.save(model.state_dict(), "./trained_model/vgg16.pth")