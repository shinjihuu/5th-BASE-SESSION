import torch.nn as nn
import torch.nn.functional as F

# ReLU 활성화, 패딩, 맥스 풀링 레이어를 포함하는 두 개의 합성곱 레이어 시퀀스 생성
def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2) 
    )
    return model

# conv_2_block과 유사하지만 세 개의 합성곱 레이어 사용
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

class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        # 여러 conv_2_block 및 conv_3_block 인스턴스 
        # 입력 이미지로부터 특징 추출
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512        
        )
        # 정규화를 위해 ReLU 활성화 및 드롭아웃과 함께 선형 레이어 사용
        # 특징 추출기에서 평평하게 만든 출력 사용 
        self.fc_layer = nn.Sequential(
            # CIFAR10은 크기가 32x32이므로 
            nn.Linear(8*base_dim*1*1, 4096),
            # IMAGENET이면 224x224이므로
            # nn.Linear(8*base_dim*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )
    # 네트워크를 통한 데이터 순전파 정의
    def forward(self, x):
        x = self.feature(x)
        #print(x.shape)
        x = x.view(x.size(0), -1) # 추출된 특징을 1D 벡터로 재구성
        #print(x.shape)
        x = self.fc_layer(x)
        return x # 클래스 확률을 나타내는 최종 출력 반환