import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler
import torchvision.transforms as T

import numpy as np
import time
import os
import cv2


DIR_TRAIN = "./data/train/"
DIR_VALID = "./data/valid/"
DIR_TEST = "./data/test/"

classes = os.listdir(DIR_TRAIN)

train_imgs = []
valid_imgs = []
test_imgs = []

for _class in classes:
    
    for img in os.listdir(DIR_TRAIN + _class):
        train_imgs.append(DIR_TRAIN + _class + "/" + img)
    
    for img in os.listdir(DIR_VALID + _class):
        valid_imgs.append(DIR_VALID + _class + "/" + img)
        
    for img in os.listdir(DIR_TEST + _class):
        test_imgs.append(DIR_TEST + _class + "/" + img)

class_to_int = {classes[i] : i for i in range(len(classes))}

def get_transform():
    return T.Compose([T.ToTensor(),
                         T.Resize((224,224))])

class BirdDataset(Dataset):
    def __init__(self, imgs_list, class_to_int, transforms=None):
        super().__init__()
        self.imgs_list = imgs_list
        self.class_to_int = class_to_int
        self.transforms = transforms

    def __getitem__(self,index):
        image_path = self.imgs_list[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        label = image_path.split("/")[-2]
        label = self.class_to_int[label]

        if self.transforms:
            image = self.transforms(image)

        return image,label

    def __len__(self):
        return len(self.imgs_list)



batch_size = 16
train_dataset = BirdDataset(train_imgs,class_to_int,get_transform())
valid_dataset = BirdDataset(valid_imgs,class_to_int,get_transform())
test_dataset = BirdDataset(test_imgs,class_to_int,get_transform())

# data loader using sampler
train_random_sampler = RandomSampler(train_dataset)
valid_random_sampler = RandomSampler(valid_dataset)
test_random_sampler = RandomSampler(test_dataset)

train_data_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, sampler = train_random_sampler)
valid_data_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, sampler = valid_random_sampler)
test_data_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, sampler = test_random_sampler)