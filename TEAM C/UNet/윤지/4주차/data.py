import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 자신의 폴더 경로에 맞게 재지정해주세요.
root_path = './datasets2/cityscapes_data/'

data_dir = root_path

# data_dir의 경로(문자열)와 train(문자열)을 결합해서 train_dir(train 폴더의 경로)에 저장합니다.
train_dir = os.path.join(data_dir, "train_or")

# data_dir의 경로(문자열)와 val(문자열)을 결합해서 val_dir(val 폴더의 경로)에 저장합니다.
val_dir = os.path.join(data_dir, "val_or")

# train_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 train_fns에 저장합니다.
train_fns = os.listdir(train_dir)

# val_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 val_fns에 저장합니다.
val_fns = os.listdir(val_dir)

sample_image_fp = os.path.join(train_dir, train_fns[0])
sample_image = Image.open(sample_image_fp).convert("RGB")

plt.imshow(sample_image)
plt.show()
# 이전에 샘플이미지에서 볼 수 있듯이, original image와 labeled image가 연결되어 있는데 이를 분리해줍니다.
def split_image(image) :
   image = np.array(image)
   
   # 이미지의 크기가 256 x 512 였는데 이를 original image와 labeled image로 분리하기 위해 리스트로 슬라이싱 합니다.
   # 그리고 분리된 이미지를 각각 cityscape(= original image)와 label(= labeled image)에 저장합니다.
   cityscape, label = image[:, :256, :], image[:, 256:, :]
   return cityscape, label

cityscape, label = split_image(sample_image)

dir_save_train = os.path.join(root_path, 'train')
dir_save_val = os.path.join(root_path, 'val')
dir_save_test = os.path.join(root_path, 'test')

# 디렉토리 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

for i in range(24):
    image_fp = os.path.join(train_dir, train_fns[i])
    image = Image.open(image_fp).convert("RGB")
    split_image(image)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), cityscape)

for i in range(24,33):
    image_fp = os.path.join(train_dir, train_fns[i])
    image = Image.open(image_fp).convert("RGB")
    split_image(image)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), cityscape)

for i in range(33,41):
    image_fp = os.path.join(train_dir, train_fns[i])
    image = Image.open(image_fp).convert("RGB")
    split_image(image)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), cityscape)

plt.subplot(121)
plt.imshow(label)
plt.title('label')

plt.subplot(122)
plt.imshow(cityscape)
plt.title('input')

plt.show()