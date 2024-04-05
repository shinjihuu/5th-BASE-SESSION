# 필요한 라이브러리
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 
from torch.utils.tensorboard import SummaryWriter

from data_transform import Normalization, ToTensor, RandomFlip
from data_loader import Dataset
from unet import UNet
#train parameter 설정
lr = 1e-3
batch_size=5
num_epoch=50

data_dir ='./unet_week3/data' # 데이터가 저장되어 있는 디렉토리
ckpt_dir ='./unet_week3/checkpoint' # 트레이닝된 네트워크가 저장될 체크 포인트 디렉토리
log_dir='./unet_week3/ log' # 텐서보드 로그파일이 저장될.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크 학습하기

# 훈련을 위해 Transform, DataLoader 불러오기
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])


dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# 네트워크 생성하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)

#손실 함수
# Sigmoid layer + BCELoss(Binaray Classification)의 조합
# 1 or 0이 나오도록
#https://cvml.tistory.com/26
fn_loss = nn.BCEWithLogitsLoss().to(device)

#Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(),lr=lr)


num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

# 배치사이즈에 의해 나눠지는 데이터 수
num_batch_train = np.ceil(num_data_train/batch_size)
num_batch_val = np.ceil(num_data_val/batch_size)


#tensor 변수에서 numpy 변수로 transfer 함수
#batch, channel, y, x -> batch, y,x,channel 
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
#denormalize
fn_denorm = lambda x, mean, std: (x * std) + mean
# network output에 대한binary class 기준 설정
fn_class = lambda x: 1.0 * (x > 0.5)

writer_train = SummaryWriter(log_dir = os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir = os.path.join(log_dir, 'val'))


## 네트워크 저장하기

def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict(),},
               './%s/model_epoch%d.pth'% (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch





## 네트워크 학습

st_epoch = 0 # 시작 에폭 0
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(st_epoch +1, num_epoch +1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        #역전파
        optim.zero_grad()

        loss = fn_loss(output, label)
        loss.backward()

        optim.step()

        # 손실함수 계산

        loss_arr += [loss.item()]

        print(f"TRAIN: EPOCH {epoch}/{num_epoch} | BATCH{batch}/{num_batch_train}, |LOSS %.4d"%np.mean(loss_arr))

        # 라벨과 이미지, 아웃풋 영상을 텐서보드에 작성하는 코드
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
        writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
        writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
        
        # loss를 텐서보드에 작성하는 코드
    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)


    with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch,data in enumerate(loader_val,1):
                #forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # calculate loss fn
                loss = fn_loss(output,label)
                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    (epoch,num_epoch,batch,num_batch_val,np.mean(loss_arr)))
                

                # save at Tensorboard
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input,mean=0.5,std=0.5))
                output= fn_tonumpy(fn_class(output))

                writer_val.add_image('label',label,num_batch_val * (epoch-1) + batch, dataformats='NHWC')
                writer_val.add_image('input',input,num_batch_val * (epoch-1) + batch, dataformats='NHWC')
                writer_val.add_image('output',output,num_batch_val * (epoch-1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss',np.mean(loss_arr),epoch)

            # model save per epoch 50
            if epoch % 50 == 0:
                save(ckpt_dir=ckpt_dir,net=net,optim=optim,epoch=epoch)

            writer_train.close()
            writer_val.close()













