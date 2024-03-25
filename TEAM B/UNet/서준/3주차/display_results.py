import os
import numpy as np
import matplotlib.pyplot as plt

# 디렉토리 설정
result_dir = './results/numpy'

# 디렉토리 속 파일 목록을 가져옴
lst_data = os.listdir(result_dir)

# 'label', 'input', 'output' 각각으로 시작하는 파일 목록을 가져옴
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

# 파일명으로 정렬
lst_label.sort()
lst_input.sort()
lst_output.sort()

# 인덱스 설정
id = 0

# numpy를 사용해서 데이터를 가져오기(numpy 배열)
label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

## 데이터 시각화
plt.subplot(131)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(132)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')

plt.show()