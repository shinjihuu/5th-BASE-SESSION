import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset 
from vgg16 import VGG16
import numpy as np
import torch 
import torch.nn as nn 
import os 
from PIL import Image
import scipy.io as sio
 
#setting
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 100

#Dataset Class(.mat)
class MyDataset(Dataset): 
    def __init__(self,root_dir,data_key,label_key,split='train',transform=None): 
        self.root_dir = root_dir
        self.data_key = data_key
        self.label_key = label_key
        self.split = split
        self.transform = transform
        self.data, self.labels = self.load_data()  

    def load_data(self):
        data_file = os.path.join(self.root_dir, f'{self.split}_32x32.mat')
        mat = sio.loadmat(data_file) 
        
        images = mat[self.data_key]
        labels = mat[self.label_key].astype('int').squeeze()
        labels[labels == 10] = 0 
    
        images = np.transpose(images, (3, 2, 0, 1)) if images.ndim == 4 else images
        return images, labels

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx] 
        label = self.labels[idx]
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.transpose((1, 2, 0)))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data
transforms = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

svhn_test = MyDataset(root_dir='./Data/', data_key='X', label_key='y',split='test', transform=transforms)

test_loader = DataLoader(svhn_test, batch_size=batch_size)

#Train
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load('./svhn_train'))

# eval
correct = 0
total = 0

model.eval() 

with torch.no_grad():
    for i,[image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        
        _,output_index = torch.max(output,1) 

        total += label.size(0) 
        correct += (output_index==y).sum().float() 

    print("Accuracy of Test DataL {}%".format(100*correct/total))