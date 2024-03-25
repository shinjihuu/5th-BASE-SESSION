import os
import numpy as np
import matplotlib.pyplot as plt

# 결과 파일이 저장된 디렉토리 설정
result_dir = './results/numpy'

# 결과 디렉토리 내 모든 파일 리스트 가져오기
lst_data = os.listdir(result_dir)

# 레이블, 입력, 출력에 해당하는 파일만 분류
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

id = 0

# 지정된 인덱스의 레이블, 입력, 출력 데이터 로드
label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

# 입력 이미지 시각화
plt.subplot(131)
plt.imshow(input, cmap='gray')
plt.title('input')

# 레이블(정답) 이미지 시각화
plt.subplot(132)
plt.imshow(label, cmap='gray')
plt.title('label')

# 모델 출력 이미지 시각화
plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')


plt.show()
