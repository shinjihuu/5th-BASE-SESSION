import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 
from torch.utils.tensorboard import SummaryWriter
#train parameter 설정
lr = 1e-3
batch_size=5
num_epoch=100

data_dir ='./unet_week3/data' # 데이터가 저장되어 있는 디렉토리
ckpt_dir ='./unet_week3/checkpoint' # 트레이닝된 네트워크가 저장될 체크 포인트 디렉토리
log_dir='./unet_week3/ log' # 텐서보드 로그기록

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## 네트워크 구조
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # conv, BN, ReLU
        def CBR2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
            layers = []
            layers += [nn.Conv2d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                bias = bias)] # conv layer 정의

            layers += [nn.BatchNorm2d(num_features=out_channels)] # BN layer 정의
            layers += [nn.ReLU()] # ReLU layer 정의

            cbr = nn.Sequential(*layers)

            return cbr

        # contracting path / enc{stage}_{layer의 index}
        self.enc1_1 = CBR2d(in_channels=1,
                            out_channels=64)

                            
                            #kernel_size=3,stride=1,padding=1,bias=True => 함수에서 미리 정의
                            
        self.enc1_2 = CBR2d(in_channels=1,out_channels=64)  
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64,out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128,out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128,out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256,out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256,out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512,out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2) # 가장 아래 stage까지 내려옴.

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path

        self.dec5_1  = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
                                          
        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512) # 스킵커넥션
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2*256, out_channels=256) # 스킵커넥션
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2*128, out_channels=128) # 스킵커넥션
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2*64, out_channels=64) # 스킵커넥션
        self.dec2_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self,x): # x는 인풋이미지
            
        enc1_1=self.enc1_1(x)
        enc1_2=self.enc1_2(enc1_1)
        pool1=self.pool1(enc1_2)
        
        enc2_1=self.enc2_1(pool1)
        enc2_2=self.enc2_2(enc2_1)
        pool2=self.pool2(enc2_2)
        
        enc3_1=self.enc3_1(pool2)
        enc3_2=self.enc3_2(enc3_1)
        pool3=self.pool3(enc3_2)
        
        enc4_1=self.enc4_1(pool3)
        enc4_2=self.enc4_2(enc4_1)
        pool4=self.pool4(enc4_2)
        
        enc5_1=self.enc5_1(pool4)
        
        dec5_1= self.dec5_1(enc5_1)

        #### 가장 하위 stage에서 다음 스테이지까지의 과정
        unpool4=self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim = 1) # 채널방향으로 concat => cat이라고 한다~
        # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x



# 데이터 로더를 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) # listdir : 데이타 디렉토리의 모든 파일 불러움

        lst_label = [f for f in lst_data if f.startswith('label')] # pre fixed 되어있는 데이터만 부름
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort() # 정렬
        lst_input.sort()

        self.lst_label = lst_label # 정렬된 리스트를 파라미터로 지정하도록 함.
        self.lst_input = lst_input
    
    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0 # 0~1로 정규화
        input = input/255.0

        if label.ndim == 2: # 채널이 없는 경우, 채널에 해당하는 축을 임의로 생성해야함.
            label = label[:,:,np.newaxis]
        if input.ndim == 2: 
            input = input[:,:,np.newaxis] # 없던 축 생성

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)
            
        return data



## 트랜스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) # numpy (x, y, ch) , pytorch (ch, y, x)
        input = input.transpose((2, 0, 1)).astype(np.float32) # numpy (x, y, ch) , pytorch (ch, y, x)

        return data

class Normalization(object):
    def __init__(self, mean = 0.5, std = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label) # flip left right
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label) # flip up down
            input = np.flipud(input)    

        data = {'label' : label, 'input' : input}

        return data        


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


if __name__ == '__main__':
    train()














