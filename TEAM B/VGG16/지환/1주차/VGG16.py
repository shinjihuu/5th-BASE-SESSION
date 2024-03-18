### X:AI 1주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch.nn as nn
import torch.nn.functional as F

# Conv layer 2 block
def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2) 
    )
    return model

# Conv layer 3 block
def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG16(nn.Module): # Module 상속
    def __init__(self, base_dim, num_classes=10): # cifar100인 경우 num_classes = 100
        super(VGG16, self).__init__()

        # image에서 특징 추출
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim), # 64
            conv_2_block(base_dim,2*base_dim), # 128
            conv_3_block(2*base_dim,4*base_dim), # 256
            conv_3_block(4*base_dim,8*base_dim), # 512
            conv_3_block(8*base_dim,8*base_dim), # 512        
        )
        
        # Fully Connected layer
        self.fc_layer = nn.Sequential( 
            nn.Linear(8*base_dim*1*1, 4096), # Flatten size = 512 x 1 x 1
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    # 순전파 함수
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layer(x)
        # probas = F.softmax(x, dim=1)
        return x