import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir_data = './unet_week3/data'

name_label = 'train-labels.tif' #512x512x30 
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data,name_label))
img_input = Image.open(os.path.join(dir_data,name_input))

#데이터 전처리
ny,nx = img_label.size #512,512
nframe = img_label.n_frames #number of 30 
# frame dmf 24, 3, 3개씩 분류 후, train, test, val에 저장
nframe_train=24
nframe_val=3
nframe_test=3

#데이터 이 저장될 디렉토리
dir_save_train = os.path.join(dir_data,'train')
dir_save_val = os.path.join(dir_data,'val')
dir_save_test = os.path.join(dir_data, 'test')

# train, val, test 디렉토리 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# 랜덤하게 이미지 추출
id_frame = np.arange(nframe) # frame에 대해 랜덤 인덱스 생성
np.random.shuffle(id_frame) 

# 실제로 저장하는 코드 작성
#train data를 저장하는 구문
#id_frame에서 0번째 id부터 시작
offset_nframe = 0

for i in range(nframe_train): # 0~23
    img_label.seek(id_frame[i+offset_nframe])
    img_input.seek(id_frame[i+offset_nframe])
    
    label_=np.asarray(img_label)
    input_=np.asarray(img_input)
    
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

#val data를 저장하는 구문
offset_nframe += nframe_train #24~26 인덱스

for i in range(nframe_val): #3번 반복
    img_label.seek(id_frame[i+offset_nframe]) #0+24, 1+24, 2+24 -> 24,25,26 data 들어감 
    img_input.seek(id_frame[i+offset_nframe])
    
    #array(원본을 객체에 할당했을 경우, 원본 변경시 객체와 별개) vs asarray(원본 변경시 객체도 변경됨)
    # https://ok-lab.tistory.com/179
    label_=np.asarray(img_label)
    input_=np.asarray(img_input)
    
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

#test data를 저장하는 구문
offset_nframe += nframe_val # 27~29 인덱스

for i in range(nframe_test): #3번 반복
    img_label.seek(id_frame[i+offset_nframe]) 
    img_input.seek(id_frame[i+offset_nframe])
    
    label_=np.asarray(img_label)
    input_=np.asarray(img_input)
    
    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)


##출력##
plt.subplot(121)
plt.imshow(label_, cmap = 'gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap = 'gray')
plt.title('input')

plt.show()
