import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from Vgg import VGG16

import argparse

# device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(save_path):
    # hyperparameter setting
    batch_size = 256
    learning_rate = 0.00015
    num_epoch = 100

    # train setting
    model = VGG16(init_dim=64).to(device)
    loss_fuction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # data pre-processing
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR100 train, test 데이터 정의
    cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, target_transform=None, download=True)
    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)

    loss_lst = []
    min_loss = np.inf
    
    for i in tqdm(range(num_epoch)):
        for j, [image,label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fuction(output, y_)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            loss_lst.append(current_loss)

        if current_loss < min_loss:
            min_loss = current_loss
            torch.save(model.state_dict(), f'{save_path}/VGG16_{i}.pth')

    plt.plot(loss_lst)
    plt.savefig(f'{save_path}/loss_plot.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGG16 model save path")
    parser.add_argument("--save_path", type=str, required=True, help="Where save .pth file")
    
    args = parser.parse_args()

    train(args.save_path)