## 라이브러리 추가하기

import argparse

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

# Parser 생성하기

parser = argparse.ArgumentParser(description='Train the UNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=1e-3, type=float, dest='lr')
parser.add_argument('--batch_size', default=4, type=int, dest='batch_size')
parser.add_argument('--num_epoch', default=100, type=int, dest='num_epoch')

parser.add_argument('--data_dir', default='./datasets', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, dest='ckpt_dir')
parser.add_argument('--log_dir', default='./log', type=str, dest='log_dir')
parser.add_argument('--result_dir', default='./results', type=str, dest='result_dir')

parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')

args = parser.parse_args()

## 트레이닝 파라미터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
# 훈련된 네트워크 저장될 checkpoint 디렉토리
log_dir = args.log_dir
# 텐서보드 로그 파일 디렉토리
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 디바이스 설정

#디렉토리 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습

if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    # 총 세가지 트랜스폼 적용시켜 데이터 로드

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # train data

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    # validation data

    # 부수적인 variable 설정
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    # 총 세가지 트랜스폼 적용시켜 데이터 로드

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    # 부수적인 variable 설정
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)


## 네트워크 생성

net = UNet().to(device)
# 네트워크가 학습되는 도메인이 CPU 기반인지, GPU 기반인지 명시


# loss function
fn_loss = nn.BCEWithLogitsLoss().to(device)

# optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 부수적인 function 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
# denormalization 함수

fn_class = lambda x: 1.0 * (x > 0.5)
# output 이미지를 binary 클래스로 분류해주는 함수

## tensorboard 사용을 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


## 네트워크 학습시키기

st_epoch = 0
# 트레이닝이 시작되는 epoch position 을 0 으로 설정

if mode == 'train':
    # train mode
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        # 네트워크 학습 이전에 저장돼있는 네트워크 있다면, 로드 후 연속적으로 네트워크 학습시킬수 있게 구현

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

            print('TRAIN: EPOCH %04d / %04d | BATCH: EPOCH %04d / %04d | LOSS %.4f' %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)


    ## Network validation 하는 부분
    # validation - back propagation 영역이 없기 때문에, torch.no_grad 로 사전에 방지

        with torch.no_grad():
            net.eval()
            # validation 위해 eval function 사용
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # loss function 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print('VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f' %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 50 == 0:
            # epoch 50 회 수행할때마다 해당 네트워크 저장
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# test mode
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    with torch.no_grad():
        net.eval()
        # validation 위해 eval function 사용
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # loss function 계산
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print('TEST: BATCH %04d / %04d | LOSS %.4f' %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print('AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f' %
          (batch, num_batch_test, np.mean(loss_arr)))
