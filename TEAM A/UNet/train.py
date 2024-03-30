## 패키지 로드
import argparse

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import Unet  # Model Architecture
from dataset import *  # Dataset & Transform
from util import *  # Save & Load

## Parser 생성하기
parser = argparse.ArgumentParser(description = "Train the Unet",
                                formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default = 1e-3, type = float, dest = 'lr')
parser.add_argument('--batch_size', default = 2, type = int, dest = 'batch_size')
parser.add_argument('--num_epoch', default = 100, type = int, dest = 'num_epoch')
parser.add_argument('--data_dir', default = '../chatbot/U-net/datasets', type = str, dest = 'data_dir')
parser.add_argument('--ckpt_dir', default = './checkpoint', type = str, dest = 'ckpt_dir')
parser.add_argument('--log_dir', default = './log', type = str, dest = 'log_dir')
parser.add_argument('--result_dir', default = './results', type = str, dest = 'result_dir')
parser.add_argument('--mode', default = 'train', type = str, dest = 'mode')
parser.add_argument('--train_continue', default = 'off', type = str, dest = 'train_continue')

args = parser.parse_args()

## 파라미터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

if mode == 'train':
    ## 네트워크 학습하기
    transform = transforms.Compose([Normalization(mean = 0.5, std = 0.5), RandomFlip(), ToTensor()])

    dataset_train = Dataset(data_dir = os.path.join(data_dir, 'train'), transform = transform)
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 8)

    dataset_val = Dataset(data_dir = os.path.join(data_dir, 'val'), transform = transform)
    loader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = False, num_workers = 8)
    ## 그밖의 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    ## 네트워크 학습하기
    transform = transforms.Compose([Normalization(mean = 0.5, std = 0.5), ToTensor()])

    dataset_test = Dataset(data_dir = os.path.join(data_dir, 'train'), transform = transform)
    loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 8)
    
    ## 그밖의 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)
    
## 네트워크 생성하기
net = Unet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr = lr)

## 그밖의 부수적인 variables 설정하기
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

## 그밖의 부수적인 functions 설정하기
fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std : (x * std) + mean
fn_class = lambda x : 1.0 * (x > 0.5)

## Tensorboard를 사용하기 위한 SummaryWriter 설정하기
writer_train = SummaryWriter(log_dir = os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir = os.path.join(log_dir, 'val'))

## 네트워크 학습시키기 -- 여기서부터 진짜 Train 과정 --
st_epoch = 0

## Train Mode
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir = ckpt_dir, net = net, optim = optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]

            print("Train : Epoch %04d / %04d | Batch %04d / %04d | Loss %.4f" %
                (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean = 0.5, std = 0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats = 'NHWC')
            writer_train.add_image('input', label, num_batch_train * (epoch - 1) + batch, dataformats = 'NHWC')
            writer_train.add_image('output', label, num_batch_train * (epoch - 1) + batch, dataformats = 'NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산
                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                print("Train : Epoch %04d / %04d | Batch %04d / %04d | Loss %.4f" %
                (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

                # Tensorboard 저장
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean = 0.5, std = 0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats = 'NHWC')
                writer_val.add_image('input', label, num_batch_train * (epoch - 1) + batch, dataformats = 'NHWC')
                writer_val.add_image('output', label, num_batch_train * (epoch - 1) + batch, dataformats = 'NHWC')
        
        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 10 == 0:
            save(ckpt_dir = ckpt_dir, net = net, optim = optim, epoch = epoch)

    writer_train.close()
    writer_val.close()

## Test Mode
else:
    net, optim, st_epoch = load(ckpt_dir = ckpt_dir, net = net, optim = optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print("Test : Batch %04d / %04d | Loss %.4f" %
            (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap = 'gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap = 'gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap = 'gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), output[j].squeeze())

                print("Average Test : Batch %04d / %04d | Loss %.4f" %
                        (batch, num_batch_test, np.mean(loss_arr)))