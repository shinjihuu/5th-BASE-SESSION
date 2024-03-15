import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


batch_size = 100 # 미니 배치사이즈를 100으로 지정
learning_rate = 0.0002 # learning rate를 0.0002로 지정
num_epoch = 100 # 전체 train data를 한바퀴(epoch)도는 횟수를 100번으로 지정

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda를 사용한다. 없으면 cpu사용
print(device) # devcie가 뭐로 지정되었는지 확인

#data
transforms = transforms.Compose( # transformor augment data for training or inference of different tasks
    [transforms.ToTensor(), # python image library(PIL) with a pixel range of [0,255]를 Pytorch Tensor(C,H,W) with a range [0.0,1.0]로 변환한다.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # 자동적으로 이미지를 정규화한다. Normalization( 평균, 표준편차 ), ([r,g,b],[r,g,b])
)

cifar10_train = datasets.CIFAR10(root = './Data/', train = True, transform = transforms, target_transform = None, download = True)
# CIFAR10의 train데이터셋을 다운로드 받는다. 경로는 /Data/. transform은 앞서 지정한대로 지정. y_train은 변환하지 않음.
cifar10_test = datasets.CIFAR10(root = './Data/', train = False, transform = transforms, target_transform = None, download = True)
# 위와 동일하지만 train이 아닌 test 데이터셋을 다운받는다. 나머지는 동일

train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle = True)
# train_loader에 데이터를 불러오는 모듈인 DataLoader를 지정한다.
# train데이터를 불러와서 batch size를 지정된 값으로 지정한 다음, 섞는다.
test_loader = DataLoader(cifar10_test, batch_size = batch_size)
# test는 shuffle하지 않는 이유???

# train
model = VGG16(base_dim = 64).to(device)
# VGG16 class를 intance화 한다.  base dim = 64 : 아키텍처 디자인 .
loss_func = nn.CrossEntropyLoss() # loss fuction으로 CE를 사용하겠다.
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# Adam optimizer를 사용.
# 반복해서 찾을 최적화하거나 찾을 파라미터

loss_arr = [] # loss array. loss를 기록하고자 빈 리스트를 만듦.
#model.train()

for i in range(num_epoch): # 사전에 지정한 epoch만큼 학습하겠다.
    for j, [image, label] in enumerate(train_loader): # train loader에서
        # j : index, image : image, label : label 을 반복문으로 받는다.
        x = image.to(device) # image를 device에서 처리하겠다.
        y = label.to(device) # label을 device에서 처리하겠다.

        optimizer.zero_grad() # optimizer의 gradient를 0으로 초기화하겠다.
        #pytorch에서는 gradients값들을 추후에 역전파시 계속 더해주기 때문에 실시한다.
        # 매번 gradient를 더해주는 방식이 기본값이다. (파이토치의 역전파)
        # 그래서 zero_grad를 하지 않으면 이상한 방향으로 향할 가능성이 높다.
        # RNN에서는 효과적이다.
        output = model.forward(x) 
        # output변수에 model의 

        loss = loss_func(output, y) # nn.CrossEntropyLoss(model.forward(x), label.to(device))
        # loss function에 model의 순전파 결과와 label을 입력하여 CE값을 구하고자함. 
        loss.backward() # 예측 손실을 역전파한다.
        optimizer.step() # current gradient를 바탕으로 파라미터를 업데이트한다.
        

    if i % 10 == 0: # 10번 epoch이 돌면 중간과정을 프린트하고자 한다.
        print(f'epoch {i} loss :', loss) # f string으로 프린트.
        loss_arr.append(loss.detach().cpu().numpy()) # loss arr라는 리스트에 추가.
            # .detach() : 연산 기록으로부터 분리한 tensor를 반환.
            # .cpu() : gpu메모리에 올려진 tensor를 cpu메모리로 복사하는 method
            # .numpy() : cpu메모리에 있는 tensor만 numpy() method를 사용가능하다.

            # 순서? : .cpu().detach() 이 순서(원래)로 진행하면 gradient 계산이 기록되는 graph에 cpu를 만드는 edge가 추가로 생성되므로 불필요한 작업을 없애려면 이 순서대로 진행해야.
            # pytorch graph ? : 동적 계산그래프. DAG.
        



torch.save(model.state_dict(), "./VGG16_100.pth")
    # state dict ? : 각 계층을 매개변수 텐서로 매핑되는 dictionary 객체이다.
    # 쉽게 저장,갱신,수정,되살리기가 가능하다. -> 모듈성을 제공
    # state dict의 항목 : 학습 가능한 매개변수를 갖는 계층, 등록된 버퍼
    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html