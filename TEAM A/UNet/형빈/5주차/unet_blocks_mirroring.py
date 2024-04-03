'''UNet Conv Blocks'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # conv batchnorm relu conv batchnorm relu
    def __init__(self,in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.double_conv(x)
    
class Down(nn.Module):
    # Down scaling with maxpool then double conv
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    # Up scaling then double conv
    def __init__(self,in_channels,out_channels,bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = DoubleConv(in_channels,out_channels,in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels ,in_channels//2,kernel_size=2,stride=2)
            self.conv = DoubleConv(in_channels,out_channels)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        ori_len = x2.size()[2]

        if diff % 2 == 0:
            x2 = x2[:,:, diff//2 : ori_len-diff//2 , diff//2 : ori_len- diff//2 ]
        
        else:
            x2 = x2[:,:, diff//2 : ori_len-(diff//2) - 1  , diff//2 : ori_len- (diff//2) -1]
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        return self.conv(x)