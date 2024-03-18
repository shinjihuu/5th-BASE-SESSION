import torch.nn as nn 
import torch.nn.functional as F 


# conv 2층짜리 
def conv_2_block(in_dim,out_dim):
    model = nn.Sequential( #torch에서, 순차적으로 레이어를 쌓는 함수 
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1), #kernel size & padding 
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_sze=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model
    
# conv 3층 짜리 
def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model
    
class VGG16(nn.Module):
    def __init__(self,base_dim, num_classes):
        super(VGG16, self).__init__()
        self.feature = nn.Sequential( #Feature map 을 추출하는 conv layer 구현 
            conv_2_block(3,base_dim), # input dim 3 for R,G,B
            conv_2_block(base_dim,base_dim*2),
            conv_3_block(base_dim*2,base_dim*4),
            conv_3_block(base_dim*4,base_dim*8),
            conv_3_block(base_dim*8,base_dim*8)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim,4096),
            nn.ReLU(), #inplace 연산 미수행 , 이전의 계산결과를 유지 
            nn.Linear(4096,4096),
            nn.ReLU(True), #inplace 연산 수행
            nn.Dropout(),
            nn.Linear(4096,1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000,num_classes),
        )
        
    def forward(self,x):
        x = self.feature()  # 정의한 모델의 feature extracting 하는 Layer(conv block) 에 전달 
        x = x.view(x.size(0),1) #flatten 
        x = self.fc_layer(x) # 정의한 모델의 Fc layer 에 전달 
        probas = F.softmax(x,dim=1)
        return probas 
        
        