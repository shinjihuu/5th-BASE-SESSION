import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 데이터 불러오기
dir_data = './datasets' 

name_label = 'train-labels.tif'  
name_input = 'train-volume.tif'  

# 이미지 파일 열기
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

# 이미지 크기 및 프레임 수 확인
ny, nx = img_label.size  # 이미지의 너비와 높이
nframe = img_label.n_frames  # 이미지 파일 내 프레임(이미지)의 총 개수

# train, val, test 세트로 나눌 이미지 프레임 수 설정
nframe_train = 24
nframe_val = 3
nframe_test = 3

# 저장할 디렉터리 경로 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 디렉터리가 존재하지 않으면 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# 이미지 프레임 순서를 무작위로 섞음
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# train 데이터 저장
offset_nframe = 0 
for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])  # 라벨 이미지 위치 조정
    img_input.seek(id_frame[i + offset_nframe])  # 입력 이미지 위치 조정

    # 이미지를 배열로 변환
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    # 배열을 파일로 저장
    np.save(os.path.join(dir_save_train, f'label_{i:03d}.npy'), label_)
    np.save(os.path.join(dir_save_train, f'input_{i:03d}.npy'), input_)

# val 데이터 저장
offset_nframe += nframe_train  
for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, f'label_{i:03d}.npy'), label_)
    np.save(os.path.join(dir_save_val, f'input_{i:03d}.npy'), input_)

# test 데이터 저장
offset_nframe += nframe_val  
for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, f'label_{i:03d}.npy'), label_)
    np.save(os.path.join(dir_save_test, f'input_{i:03d}.npy'), input_)

# 마지막으로 저장된 데이터 시각화
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('Label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('Input')

plt.show()
