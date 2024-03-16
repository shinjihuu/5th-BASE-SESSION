# VGG16 Model Architecture
# 16-Layer = 13 Convolution Layer + 3 Fully-Connected Layer
# 3x3 convolution filter, stride = 1
# 2x2 max pooling, stride = 2
# ReLU
# 224x224x3 (RGB) 이미지를 input으로 받음

# 3x3 합성곱 연산 x2 (채널 64)
# 3x3 합성곱 연산 x2 (채널 128)
# 3x3 합성곱 연산 x3 (채널 256)
# 3x3 합성곱 연산 x3 (채널 512)
# 3x3 합성곱 연산 x3 (채널 512)
# FC layer x3 : 4096, 4096, 1000

import torch
import torch.nn as nn
import torch.nn.functional as F

# conv layer가 2개 있는 block과 3개 있는 block을 따로 선언
# 합성곱 2 + Max Pooling 1
def conv_2_block(input_dim, output_dim):
    model = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1), # 3x3 커널, padding=1을 설정하여 입력 이미지의 크기가 줄어들지 않도록
        nn.ReLU(), # 활성화 함수 -> 비선형성
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1), # 3x3 커널
        nn.ReLU(), # 활성화 함수
        nn.MaxPool2d(2,2) # 2x2 MaxPooling, 이미지 크기를 반으로 줄임
    )
    return model

# 합성곱 3 + Max Pooling 1, 더 복잡한 특성을 추출하는 데 사용
def conv_3_block(input_dim, output_dim):
    model = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2) # MaxPooling 적용하여 차원 축소
    )
    return model

# VGG16 모델 클래스
class VGG16(nn.Module):
    def __init__(self, base_dim, num_classes=10): # 기본 차원 수, 분류할 클래스 수
        super(VGG16, self).__init__()
        # 피처 추출
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), # 첫 번째 블록 (입력 채널 3)
            conv_2_block(base_dim, base_dim*2), # 두 번째 블록 (차원 증가)
            conv_3_block(base_dim*2, base_dim*4), # 세 번째 블록 (차원 증가)
            conv_3_block(base_dim*4, base_dim*8), # 네 번째 블록 (차원 증가)
            conv_3_block(base_dim*8, base_dim*8), # 다섯 번째 블록 (동일 차원)
        )
        # 분류를 위한 FC층
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 4096), # 피처 맵 평탄화 -> 첫 번째 FC층
            nn.ReLU(True), # 활성화 함수로 ReLU 사용, 비선형성 추가 / True 넣으면?
            nn.Dropout(), # 과적합 방지를 위한 드롭아웃
            nn.Linear(4096, 1000), # 두 번째 FC층, 4096개의 노드에서 1000개의 노드로 줄임
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes), # 최종 출력 층, 1000개의 노드에서 클래스 수에 해당하는 노드로 연결
        )

    def forward(self, x):
        x = self.feature(x) # 특성 추출 
        x = x.view(x.size(0), -1) # 배치 차원을 유지하면서 나머지 차원을 평탄화
        x = self.fc_layer(x) # FC층을 통과
        # 소프트맥스 추가
        return x
    
# references
# https://velog.io/@euisuk-chung/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EB%A1%9C-CNN-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EC%9E%90-VGGNet%ED%8E%B8
# https://beginnerdeveloper-lit.tistory.com/158
# https://velog.io/@euisuk-chung/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EB%A1%9C-CNN-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EC%9E%90-VGGNet%ED%8E%B8