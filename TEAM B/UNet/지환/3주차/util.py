### X:AI 3주차 Code 과제
### AI빅데이터융합경영 배지환 

import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
# 모델 가중치와 optimzer load
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir) # checkpoint 파일 목록 list
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # 파일 이름에 포함된 숫자를 기준으로 정렬

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1])) # 최신 checkpoint 파일 load / 숫자를 기준으로 정렬했으니 가장 끝이 최신

    # 불러온 가중치, optim으로 업데이트
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    # 최신 checkpoint 파일에서 epoch 번호 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch