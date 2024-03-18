import torch
from tqdm import tqdm
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from MyVGG16 import MyVGG

# setting
batch_size = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

# data loader
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5]), 
                                transforms.Resize((32,32))]) 

test_dset = datasets.CIFAR10(root="./data/", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dset, batch_size=batch_size)

model = torch.load('VGG_model_01.pth')
# model = MyVGG(base_channel_dim=64).to(device)
# model.load_state_dict(torch.load('VGG_model_01.pth'))

correct = 0
total = 0

# inference
model.eval()

# 인퍼런스 모드 :  no_grad 
with torch.no_grad():
    # 테스트로더에서 이미지와 라벨 불러와서
    for image,label in tqdm(test_loader, total= batch_size):
        x = image.to(device)
        y = label.to(device)

        # 모델에 데이터 넣고 결과값 얻기
        output = model.forward(x)
        _,output_index = torch.max(output,1)
        # torch.max() : tensor에서 최댓값과 해당 최댓값의 인덱스를 반환
        # 1은 찾고자 하는 최댓값을 찾을 차원을 지정하는 매개변수
        # 1 : 각 행에서 가장 큰 값과 그 인덱스를 반환

        # 전체 개수 += 라벨의 개수
        total += label.size(0)
        correct += (output_index == y).sum().float()
    
    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))
    
    model.eval()
