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

from model import UNet # network models
from dataset import * # Dataset & Transform
from util import * # Network's save & load


## Parser 생성하기
# parser object생성
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser에 argument 추가
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")# Learning rate받는 parser
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

# 저장 디렉토리 추가
parser.add_argument("--data_dir", default='.\code-unet\datasets', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='.\code-unet\checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='.\code-unet\log', type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

# train test mode 구분
parser.add_argument("--mode", default="train", type=str, dest="mode")
# train을 앞서 저장한 네트워크 불러와서 추가 트레이닝 시킬지 여부
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()


# 하이퍼파라미터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
# train된 네트워크가 저장될 위치
ckpt_dir = args.ckpt_dir
# 텐서보드 log파일 저장될 위치
log_dir = args.log_dir
# 결과를 저장할 디렉토리 경로 설정
result_dir = args.result_dir
#result_dir = os.path.abspath('.\code-unet\Results')
mode = args.__module__
train_continue = args.train_continue

# 결과를 저장할 디렉토리가 존재하지 않으면 생성
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 'png' 디렉토리 생성
png_dir = os.path.join(result_dir, 'png')
if not os.path.exists(png_dir):
    os.makedirs(png_dir)
# 'numpy' 디렉토리 생성
numpy_dir = os.path.join(result_dir, 'numpy')
if not os.path.exists(numpy_dir):
    os.makedirs(numpy_dir)

#gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## 네트워크 학습하기
if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    # Compose: list of transforms to compose
    # 데이터 불러올 때 위의 transform 같이 적용시켜 불러옴

    # 폴더로부터 필요한 데이터 불러옴, DataLoader
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    ## 부수적인 variables 생성
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    # 배치사이즈에 의해 나누어지는 데이터셋의 수
    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    # Compose: list of transforms to compose
    # 데이터 불러올 때 위의 transform 같이 적용시켜 불러옴

    # 폴더로부터 필요한 데이터 불러옴, DataLoader
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    ## 부수적인 variables 생성
    num_data_test = len(dataset_test)

    # 배치사이즈에 의해 나누어지는 데이터셋의 수
    num_batch_test = np.ceil(num_data_test / batch_size)


## 네트워크 생성
net = UNet().to(device)

# 손실함수 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)
# Optimizer 생성
optim = torch.optim.Adam(net.parameters(), lr=lr)
## 부수적인 function 생성
# 텐서에서 넘파이로 변환
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
# denormalization 함수
fn_denorm = lambda x, mean, std: (x*std) + mean
# 네트워크 아웃풋의 이미지를 바이너리클래스로 분류
fn_class = lambda x: 1.0* (x>0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


## 네트워크 학습, 실제 트레이닝 수행되는 for loop 구현
st_epoch = 0

#TRAIN MODE
if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    
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

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # 네트워크 validation
        with torch.no_grad(): #backpropagation 없으므로 사전에 torch.no_grad()
            net.eval() # val설정
            loss_arr = []
            
            # forward pass
            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)
            
                # loss 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
        #loss function 저장
        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
        
        # epoch  진행 마다 네트워크 저장
        if epoch//50==0:#50번마다 저장
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
# TEST MODE
else:
    # 저장된 네트워크 로드
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    # 네트워크 validation
    with torch.no_grad(): #backpropagation 없으므로 사전에 torch.no_grad()
        net.eval() # val설정
        loss_arr = []

        # forward pass
        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)
            
            # loss 계산
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                ( batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))
            
            # 각각의 슬라이스 따로 저장
            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j
                # png파일로 저장
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')
                
                #numpy파일로 저장
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    # 최종 테스트셋의 평균 손실함수값           
    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
        (batch, num_batch_test, np.mean(loss_arr)))    

            
    