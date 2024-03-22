import torchvision.datasets as datasets
import torchvision.datasets as MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from VGG16 import vgg16
from Custom import CustomMNIST

batch_size = 30
learning_rate = 0.0002
num_epoch = 10 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device) # devcie가 뭐로 지정되었는지 확인

MNIST_train = CustomMNIST(root = '/home/work/test/vgg16_week2/data', train = True)


train_loader = DataLoader(MNIST_train, batch_size = batch_size, shuffle = True)



# train
model = vgg16(batch_size=batch_size, base_dim = 64).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


loss_arr = [] 
model.train()

for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device) 
        y = label.to(device) 

        optimizer.zero_grad()
        output = model.forward(x) 

        loss = loss_func(output, y) 
        loss.backward() 
        optimizer.step()
        
    if i % 10 == 0: 
        print(f'epoch {i} loss :', loss) 
        loss_arr.append(loss.detach().cpu().numpy())

torch.save(model.state_dict(), "./VGG16_100.pth")