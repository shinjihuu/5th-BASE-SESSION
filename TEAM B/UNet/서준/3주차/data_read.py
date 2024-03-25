# 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 데이터 불러오기
dir_data = './datasets'

# 파일 지정
name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

# 레이블 및 입력 이미지 열기
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

# 이미지의 높이, 너비, 프레임 수 가져오기
ny, nx = img_label.size
nframe = img_label.n_frames

# train, validation, test할 때 사용할 프레임 수
nframe_train = 24
nframe_val = 3
nframe_test = 3

# 각각의 데이터 저장 디렉토리 경로 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 저장할 디렉토리가 없다면 새로 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# 이미지 프레임 수 만큼 인덱스 생성하고 섞기
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# 프레임의 offset 설정
offset_nframe = 0

for i in range(nframe_train):
    # 파일의 출력 시작점 설정
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    # array : copy = True
    # asarray : copy = False
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    # .npy 확장자로 저장
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# 프레임의 offset 설정
offset_nframe = nframe_train

for i in range(nframe_val):
    
    # 파일의 출력 시작점 설정
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    # array : copy = True
    # asarray : copy = False
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    
    # .npy 확장자로 저장
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

# 프레임의 offset 설정
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 데이터 시각화 (matplotlib 사용)
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()