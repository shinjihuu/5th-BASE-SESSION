import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

#setting
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 100

# Data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
'''
cifar10_test = datasets.CIFAR10(root = "./Data/", train = False, transform = transforms, target_transform = None, download = True)

test_loader = DataLoader(cifar10_test, batch_size = batch_size)
'''

#Train

model = VGG16(base_dim = 64).to(device)
# 여기까지는 train과 동일함.

model.load_state_dict(torch.load('./VGG16_100.pth'))
# trian에서 학습했던 모델의 파라미터들을 불러온다. load_state_dict.


#eval

correct = 0 # 맞은 라벨의 개수를 입력할 변수를 초기화
total = 0 # 전체 라벨 사이즈를 입력할 변수를 초기화


model.eval()
# 모델을 평가모드로 전환하는 메서드이다.
# 별도의 인수를 받지 않는다.
# 평가모드로 전환된 모델은 dropout이 비활성화, batch normalization의 평균과 이동분산이 업데이트되지 않는다.
# 학습싱만 필요했떤 기능을 비활성화 하여, 모델의 성능 평가에 불필요한 노이즈를 줄이고 일관된 결과를 얻을 수 있다.
# https://wikidocs.net/195115

with torch.no_grad():
# autograd engine(gradient를 계산해주는 condtext)를 비활성화.
# 필요한 메모리를 줄여주고 연산속도를 증가시킨다.
    for i, [image, label] in enumerate(test_loader): # test_loader를 하나씩 받는다.
        x = image.to(device) # x에는 image를
        y = label.to(device) # y에는 label을

        output = model.forward(x) 
        # output에는 model을 통과한 x의 결과물
        # (배치크기)x(클래스의 개수). 즉, 열이 하나의 이미지와 대응되는 벡터임.
        _, output_index = torch.max(output, 1) # 0(axis = 0):행, 1(axis = 1):열
        # 즉 열에서 최대값의 " 인덱스 "를 뽑아내는 모듈. torch.max()
        # torch.max(output.data, 1)이라고 작성된 경우도 있다.
        # with torch.no_grad()를 사용안했다면.
        # 예측값을 계산할 때는 역전파 계산이 필요없기 때문에 데이터만 사용한다는 의미.
        #https://www.inflearn.com/questions/282058/%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80-%EB%B6%80%EB%B6%84-%EC%A7%88%EB%AC%B8%EB%93%9C%EB%A6%BD%EB%8B%88%EB%8B%A4
        total += label.size(0)
        # label 텐서의 0번째 차원 크기를 total에 더하고자 함.
        # 즉 배치크기를 추가하는 것으로, 배치의 샘플 수만큼 증가시킨다.
        # 100개로 설정했으니 total은 100이 되겠지?
        correct += (output_index ==y).sum().float()
        # 정답이면 correct에 추가한다.

    print("Accuracy of Test Data : {}%".format(100*correct/total)) # 정확도