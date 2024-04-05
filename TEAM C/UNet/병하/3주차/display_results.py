import os
import numpy as np
import matplotlib.pyplot as plt

## 
result_dir = './results/numpy'

lst_data = os.listdir(result_dir) # 파일 이름들을 list로 반환

lst_label = [f for f in lst_data if f.startswith('label')] # label로 시작하는 파일 이름들 list로 반환
lst_input = [f for f in lst_data if f.startswith('input')] # input으로 시작하는 파일 이름들 list로 반환
lst_output = [f for f in lst_data if f.startswith('output')] # output으로 시작하는 파일 이름들 list로 반환

lst_label.sort() # 정렬
lst_input.sort()
lst_output.sort()

##
id = 0

label = np.load(os.path.join(result_dir, lst_label[id])) # result_dir에서 첫번재 label 결과 load
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

## 시각화
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