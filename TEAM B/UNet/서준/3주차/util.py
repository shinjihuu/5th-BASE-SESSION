import os
import numpy as np
import torch
import torch.nn as nn

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    
    # 만약 ekpt_dir 폴더가 없다면
    if not os.path.exists(ckpt_dir):
        # ekpt_dir 폴더 생성
        os.makedirs(ckpt_dir)

    # torch 모델 저장 (%s/model_epoch%d.pth로)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
    
## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    # ckpt_dir 폴더 내에 있는것들을 불러옴
    ckpt_lst = os.listdir(ckpt_dir)

    # 불러온 ckpt_lst 리스트들을 정렬
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # 모델 불러오기
    dict_model = torch.load('%s%s' % (ckpt_dir, ckpt_lst[-1]))

    # load_state_dict() : 역직렬화된 state_dict를 사용하여 모델의 매개변수들을 불러옴
    # net, optim을 dict형태로 불러옴
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    # net, optim, epoch를 반환
    return net, optim, epoch