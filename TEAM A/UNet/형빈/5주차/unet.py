# https://www.kaggle.com/code/gokulkarthik/image-segmentation-with-unet-pytorch/notebook
# https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py
from unet_blocks_mirroring import *

## build architecture
class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels,64)

        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512,1024//factor)

        self.up1 = Up(1024,512//factor,bilinear)
        self.up2 = Up(512,256//factor,bilinear)
        self.up3 = Up(256,128//factor,bilinear)
        self.up4 = Up(128,64,bilinear)
        self.outc = OutConv(64,n_classes)
        
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        logits = self.outc(x)

        # float을 int로 변환하며 손실된 숫자들로 error의 경우가 발생함.
        # 마지막 layer를 통과한 값에 대해 손실된 값 padding으로 채움.

        if logits.size()[2] != 256:
            diff = 256-logits.size()[2]
            logits = F.pad(logits,[diff//2,diff//2,diff//2,diff//2])
        return logits




    