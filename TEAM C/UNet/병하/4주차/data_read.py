import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif' # 512x512x30
name_input = 'train-volume.tif' # 512x512x30

img_label = Image.open(os.path.join(dir_data, name_label)) 
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size # image의 x, y 수 (512개)
nframe = img_label.n_frames # frame 수 (30개)

## 30개 frame 중 24개는 train / 3개는 val / 나머지 3개는 test로 사용
nframe_train = 24
nframe_val = 3
nframe_test = 3

## 저장 경로 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

## 경로 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 랜덤 frame index 생성
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

## train set 저장
offset_nframe = 0
# 0~23
for i in range(nframe_train): # 24
    img_label.seek(id_frame[i + offset_nframe]) # i + offset_nframe에 해당하는 index로 이동
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label) # numpy 배열로 변환
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_) # 저장 경로에 저장
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

## validation set 저장
offset_nframe = nframe_train

# 24~26
for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## test set 저장
offset_nframe = nframe_train + nframe_val

# 27~29
for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 생성한 dataset 출력
# plt.subplot(121)
# plt.imshow(label_, cmap='gray')
# plt.title('label')

# plt.subplot(122)
# plt.imshow(input_, cmap='gray')
# plt.title('input')

# plt.show()