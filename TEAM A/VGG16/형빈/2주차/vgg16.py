# 3x3 conv stride : 1
# 2x2 Max pooling stride : 2
# Actfunc : ReLU

import torch
import torch.nn as nn
import torch.nn.functional as F

# 3x3 convolution x2 block

def conv2_block(in_dim, out_dim):
    model = nn.Sequential(
    nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
    nn.MaxPool2d(2,2)
        )
    return model

# 3x3 convolution x3 block 
def conv3_block(in_dim, out_dim):
    model = nn.Sequential(
    nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2)
        )
    return model


class vgg(nn.Module):
    def __init__(self, dim, num_classes=525):
        super(vgg,self).__init__()
        self.feature = nn.Sequential(
            conv2_block(3,dim), #64
            conv2_block(dim,2*dim), #128
            conv3_block(2*dim,4*dim), # 256
            conv3_block(4*dim,8*dim), # 512
            conv3_block(8*dim,8*dim) # 512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*dim*7*7, 4096),
            # MaxPool이 5번 진행되었으므로 2^5만큼 이미지 가로 세로가 줄어듬.
            # 사용할 데이터셋은 CIFAR10이고, 데이터셋의 이미지 가로세로는 32x32이므로 1*1을 함.
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048,num_classes)
        )

    def forward(self,x):
        x = self.feature(x)
        #print("x.shape",x.shape)
        x = x.view(x.size(0),-1) # 1차원으로 flatten
        #print("x.size(0)",x.size(0))
        #print("x.shape",x.shape)
        x = self.fc_layer(x)
        return x