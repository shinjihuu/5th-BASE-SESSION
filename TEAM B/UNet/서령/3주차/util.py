import os
import numpy as np

import torch
import torch.nn as nn

# 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    # 체크포인트 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 네트워크와 optimizer의 상태를 저장
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    # 체크포인트 디렉토리가 존재하지 않으면 초기 상태 반환
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    # 체크포인트 파일 목록을 불러온 후, 파일 이름에 포함된 숫자를 기준으로 정렬
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # 가장 최근 체크포인트 불러오기
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    # 네트워크와 optimizer의 상태를 불러온 정보로 업데이트
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    
    # epoch 번호 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch