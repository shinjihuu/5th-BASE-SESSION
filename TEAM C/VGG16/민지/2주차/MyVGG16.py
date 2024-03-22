import torch.nn as nn
# conv 연산 : (input - filter +2*padding)/stride + 1
# pooling 연산 : input/filter

def conv_2_block(in_channel, out_channel):
    model = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2,2) # kernel_size, stride_size => 2x2 MaxPooling layer (stride=2)
    )
    return model


def conv_3_block(in_channel, out_channel):
    model = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
    return model



class MyVGG(nn.Module):  # PyTorch의 모든 Neural Network의 Base Class
    def __init__(self,base_channel_dim=64, num_class=10):
        super(MyVGG, self).__init__()
        # super() 함수는 파이썬에서 부모 클래스의 메서드를 호출하는 데 사용
        # MyVGG 클래스의 부모 클래스인 nn.Module의 __init__() 메서드를 호출하는 것
        
        self.features = nn.Sequential(
            # (32,32,3) --> (16,16,64)        (224*224,3) --> (112,112,64)
            conv_2_block(3, base_channel_dim),
            # (16,16,64) --> (8,8,128)        (112,112,64) --> (56,56,128)
            conv_2_block(base_channel_dim, base_channel_dim*2),
            # (8,8,128) --> (4,4,256)         (56,56,128) --> (28,28,256)
            conv_3_block(base_channel_dim*2, base_channel_dim*4),
            # (4,4,256)  --> (2,2,512)        (28,28,256)  --> (14,14,512)
            conv_3_block(base_channel_dim*4, base_channel_dim*8),
            # (2,2,512) --> (1,1,512)         (14,14,512) --> (7,7,512)
            conv_3_block(base_channel_dim*8, base_channel_dim*8),
                                 )
        
        self.classifier = nn.Sequential(
            # 64*8 == 512  
            nn.Linear(base_channel_dim*1*1*8, 4096),  # 입력 이미지 크기 32*32
            # nn.Linear(base_channel_dim*7*7*8, 4096), # 입력 이미지 크기가 224,224 일 경우 (ImageNet)
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_class)
            ### 변경. 내 데이터 class 수는 10이므로!
            )
        ## 분류문제임에도 softmax를 사용하지 않는 이유
        # Pytorch의 nn.CrossEntropyLoss는 log softmax와 negative log likelihood를 결합하여 구현되어 있음
        # softmax를 사용하고 싶으면 softmax를 거친 결과에 log를 취한 후, nn.NLLLoss에 입력해야 함 (그렇게 안해도 결과는 같긴 함)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # 배치는 keep -> 열을 h*w로  [32,3,3] -> [32,9]
        x = self.classifier(x)
        return x

