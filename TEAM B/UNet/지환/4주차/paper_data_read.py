### X:AI 4주차 Code 과제
### AI빅데이터융합경영 배지환 


import os
import numpy as np
from PIL import Image



dir_data = './kaggle_datasets'

# train
for i in range(1, 21):
    name_label = f'TCGA_CS_4942_19970222_{i}_mask.tif' 
    name_input = f'TCGA_CS_4942_19970222_{i}.tif' 

    img_label = Image.open(os.path.join(dir_data, name_label)) 
    img_input = Image.open(os.path.join(dir_data, name_input))

    ny, nx = img_label.size 
    nframe = img_label.n_frames


    ## 저장 경로 설정
    dir_save_train = os.path.join(dir_data, 'train')

    ## 경로 생성
    if not os.path.exists(dir_save_train):
        os.makedirs(dir_save_train)


    label_ = np.asarray(img_label) 
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_) 
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# val
for i in range(1, 4):
    name_label = f'TCGA_CS_4943_20000902_{i}_mask.tif' 
    name_input = f'TCGA_CS_4943_20000902_{i}.tif' 

    img_label = Image.open(os.path.join(dir_data, name_label)) 
    img_input = Image.open(os.path.join(dir_data, name_input))

    ny, nx = img_label.size 
    nframe = img_label.n_frames 

    ## 저장 경로 설정
    dir_save_train = os.path.join(dir_data, 'val')

    ## 경로 생성
    if not os.path.exists(dir_save_train):
        os.makedirs(dir_save_train)


    label_ = np.asarray(img_label) 
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_) 
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# test
for i in range(3, 6):
    name_label = f'TCGA_CS_4943_20000902_{i}_mask.tif' 
    name_input = f'TCGA_CS_4943_20000902_{i}.tif' 

    img_label = Image.open(os.path.join(dir_data, name_label)) 
    img_input = Image.open(os.path.join(dir_data, name_input))

    ny, nx = img_label.size 
    nframe = img_label.n_frames 

    ## 저장 경로 설정
    dir_save_train = os.path.join(dir_data, 'test')

    ## 경로 생성
    if not os.path.exists(dir_save_train):
        os.makedirs(dir_save_train)


    label_ = np.asarray(img_label) 
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_) 
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

