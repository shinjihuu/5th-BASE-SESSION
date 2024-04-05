import numpy as numpy
import torch
import torch.nn as nn


## 네트워크 구조
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # conv, BN, ReLU == 파란색 화살표
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

        # contracting path / contract{stage}_{layer의 index}
        self.contract1_1 = CBR2d(in_channels=1, # 흑백이다.
                            out_channels=64) # 첫번째 파란색 화살표

                            
                            #kernel_size=3,stride=1,padding=1,bias=True => 함수에서 미리 정의
                            
        self.contract1_2 = CBR2d(in_channels=64,out_channels=64)  # 첫번째 layer의 두번째 파란색 화살표
        
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 첫번째 빨간색 화살표

        self.contract2_1 = CBR2d(in_channels=64,out_channels=128)
        self.contract2_2 = CBR2d(in_channels=128,out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.contract3_1 = CBR2d(in_channels=128,out_channels=256)
        self.contract3_2 = CBR2d(in_channels=256,out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.contract4_1 = CBR2d(in_channels=256,out_channels=512)
        self.contract4_2 = CBR2d(in_channels=512,out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2) # 가장 아래 stage까지 내려옴.

        self.contract5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path

        self.expansive5_1  = CBR2d(in_channels=1024, out_channels=512)

        self.upconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
                                          
        self.expansive4_2 = CBR2d(in_channels=2*512, out_channels=512) # 스킵커넥션
        self.expansive4_1 = CBR2d(in_channels=512, out_channels=256)

        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.expansive3_2 = CBR2d(in_channels=2*256, out_channels=256) # 스킵커넥션
        self.expansive3_1 = CBR2d(in_channels=256, out_channels=128)

        self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.expansive2_2 = CBR2d(in_channels=2*128, out_channels=128) # 스킵커넥션
        self.expansive2_1 = CBR2d(in_channels=128, out_channels=64)

        self.upconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.expansive1_2 = CBR2d(in_channels=2*64, out_channels=64) # 스킵커넥션
        self.expansive1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True) # 녹색화살표

    def forward(self,x): # 이제 모든 걸 다 연결해보겠다. x는 인풋이미지
            
        contract1_1=self.contract1_1(x) # (1,572,572)->(64,570,570)
        contract1_2=self.contract1_2(contract1_1) # (64,570,570)->(64,568,568)
        pool1=self.pool1(contract1_2) # (64,568,568)->(64,284,284)
        
        contract2_1=self.contract2_1(pool1) # (64,284,284)->(128,282,282)
        contract2_2=self.contract2_2(contract2_1) # (128,282,282)->(128,280,280)
        pool2=self.pool2(contract2_2) # (128,280,280)->(128,140,140)
        
        contract3_1=self.contract3_1(pool2) # (128,140,140)->(256,138,138)
        contract3_2=self.contract3_2(contract3_1) # (256,138,138)->(256,136,136)
        pool3=self.pool3(contract3_2) # (256,136,136)->(256,68,68)
        
        contract4_1=self.contract4_1(pool3) # (256,68,68)->(512,66,66)
        contract4_2=self.contract4_2(contract4_1) # (512,66,66)->(512,64,64)
        pool4=self.pool4(contract4_2) # (512,64,64)->(512,32,32)
        
        contract5_1=self.contract5_1(pool4) # (512,32,32)->(1024,30,30)
        
        expansive5_1= self.expansive5_1(contract5_1) # (1024,30,30)->(1024,28,28)

        #### 가장 하위 stage4에서 다음 스테이지까지의 과정
        upconv4=self.upconv4(expansive5_1) # (1024,28,28)->(512,56,56)

        concat4 = torch.cat((upconv4, contract4_2), dim = 1) #(512,56,56) + (512,56,56) = (1024,56,56)
        # 채널방향으로 concat => cat이라고 한다~
        # dim = [0:batch, 1:channel, 2:height(y방향), 3:width(x방향)]
        expansive4_2 = self.expansive4_2(concat4) # (1024,56,56)->(512,54,54)
        expansive4_1 = self.expansive4_1(expansive4_2) # (512,54,54)->(512,52,52)

        upconv3 = self.upconv3(expansive4_1) # (512,52,52)->(256,104,104)
        concat3 = torch.cat((upconv3, contract3_2), dim=1) # (256,104,104) + (256,104,104) = (512,104,104)
        expansive3_2 = self.expansive3_2(concat3) # (512,104,104)->(256,102,102)
        expansive3_1 = self.expansive3_1(expansive3_2) # (256,102,102)->(256,100,100)

        upconv2 = self.upconv2(expansive3_1) # (256,100,100)->(128,200,200)
        concat2 = torch.cat((upconv2, contract2_2), dim=1) # (128,200,200) + (128,200,200) = (256,200,200)
        expansive2_2 = self.expansive2_2(concat2) # (256,200,200)->(128,198,198)
        expansive2_1 = self.expansive2_1(expansive2_2) # (128,198,198)->(128,196,196)

        upconv1 = self.upconv1(expansive2_1) # (128,196,196)->(64,392,392)
        concat1 = torch.cat((upconv1, contract1_2), dim=1) # (64,392,392) + (64,392,392) = (128,392,392)
        expansive1_2 = self.expansive1_2(concat1) # (128,392,392)->(64,390,390)
        expansive1_1 = self.expansive1_1(expansive1_2) # (64,390,390)->(64,388,388)

        x = self.fc(expansive1_1) # (64,388,388)->(1,388,388)

        return x # (1, 388, 388)
