import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.notebook import tqdm

data_dir = os.path.join("./dataset","cityscapes_data")
train_dir = os.path.join(data_dir,"train")
val_dir = os.path.join(data_dir,"val")
train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)

num_classes = 30
label_model = KMeans(n_clusters=num_classes,n_init=10)

num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
label_model.fit(color_array)

class CityscapeDataset(Dataset):
    
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.label_model = label_model
        
    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):

        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        cityscape, label = self.split_image(image)
        
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256) # 색 별 라벨 지정하기 위해 kmeans 사용한듯
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label_class).long() # 정수 연산은 보통 long tesnor, 실수 연산은 보통 float tensor
        return cityscape, label_class
    
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :] # input이 가로를 기준으로 왼쪽 절반은 원본, 오른쪽 절반은 라벨임.
        return cityscape, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)
    
dataset = CityscapeDataset(train_dir, label_model)
val_dataset = CityscapeDataset(val_dir, label_model)




















