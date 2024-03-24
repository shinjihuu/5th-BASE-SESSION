import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import DogCatDataset

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from Vgg_new import VGG16

import argparse

# device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(save_path):
    # hyperparameter setting
    batch_size = 256
    learning_rate = 0.0015
    num_epoch = 10

    # train setting
    model = VGG16(init_dim=64).to(device)
    loss_fuction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Resize((128, 128)),  # 이미지 크기 조정
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR100 train, test 데이터 정의
    train_dataset = DogCatDataset(data_dir='dog_cat/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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