import torch.nn as nn
import torch.nn.functional as F # softmax를 호출하기 위해 import

def conv_2_block(in_dim, out_dim): # 입력 채널과, 출력 채널을 인수로 갖는다.
# conv filter가 두개
    model = nn.Sequential( # 이렇게 안하면 class로 하나하나 지정해야함.
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1), # 3*3, zero_padding 한바퀴 돌린 필터
        nn.ReLU(), # ReLU 활성화 함수 사용
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),# 3*3, zero_padding 한바퀴 돌린 필터
        # Conv2d 의 output의 size인 out_dim을 받는다.
        nn.ReLU(), # ReLU 활성화 함수 사용
        nn.MaxPool2d(2,2) # maxpooling을 적용한다.

    )
    return model # 우리가 nn.Sequential에서 정의한 모델을 반환.
 # nn 모듈을 불러왔다~




# conv 블록이 두개 인것 세개 인 것 확인.

def conv_3_block(in_dim, out_dim): # conv filter가 세개
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )

    return model


class vgg16(nn.Module): # nn.Sequential, nn.Linear, nn.ReLU, nn.Dropout 을 받기 위함.
    def __init__(self, base_dim,batch_size, num_classes = 10):
        super(vgg16, self).__init__()
        self.batch_size = batch_size
        self.feature = nn.Sequential( # conv 블록 정의
            conv_2_block(1, base_dim), # mnist 는 흑백이라 channel = 1
            conv_2_block(base_dim, 2*base_dim), # 64 -> 128
            conv_3_block(2*base_dim, 4*base_dim), # 128 -> 256
            conv_3_block(4*base_dim, 8*base_dim), # 256 -> 512
            conv_3_block(8*base_dim, 8*base_dim), # 512 -> 512
        )
        self.fc_layer = nn.Sequential( # FC 블록 정의
            nn.Linear(25088, 4096), # 512 , 512*2*2*2
            # 마지막 연결 conv layer를 flatten하면 7*7*512 = 25088
            # 차원을 감소시키기 위해 4096으로 지정(아키텍처에서 지정함)
            nn.ReLU(),
            # 그러면 얘는 왜 inplace = False??????????????
            # input값을 소실 시키면 원본 텐서에 손상이 간다.
            # 그래서 처음 ReLU는 False이지 않을까?
            nn.Linear(4096, 4096), # 4096 -> 4096
            nn.ReLU(True),
            # inplace = True : 인풋값에 대한 결괏값을 따로 저장하는 것이 아닌
            # 기존의 데이터 값을 대신한다. input값 대신 output값만 남음
            # 메모리 약간의 이득.
            nn.Dropout(), # 아키텍처 디자인
            nn.Linear(4096, batch_size), # 아키텍처 디자인
            nn.ReLU(True), # 아키텍처 디자인
            nn.Dropout(), # 아키텍처 디자인
            nn.Linear(batch_size, num_classes), # 100(아키텍처 디자인)개로 받아서, 최종 class 개수로 출력
        
        )
    def forward(self, x): # 순전파
        x = self.feature(x) # 입력 텐서x를 vgg16으로 보냄
        x = x.view(x.size(0), -1) # fc_layer로 보내기 전 flatten 
        x = self.fc_layer(x) # fc_layer로 보냄
        probas = F.softmax(x, dim = 1) # softmax(class별 출력을 확률 열벡터로 출력)
        return probas # softmax 열벡터를 return