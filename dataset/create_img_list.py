# -*- coding: utf-8 -*- #
'''
# ------------------------------------------------------------------------
# File Name:        WSOL_RS/dataset/creat_img_list.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/10/19
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
#                           --> 遥感图像, 弱监督目标定位项目代码 <--        
#                   -- 划分数据集(PatternNet, TODO), 生成WSOL_EVAL框架所需的
#                   数据格式, 即：图像路径 标签 bbox, 若同一图像出现多个目标，
#                   生成多行bbox说明
#                   — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    WSOL_EVAL/utils.logger
# Function List:    <0> None
# Class List:       <0> None
#                   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
#      |  <author>  | <version> |   <time>   |         <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#  <0> |    rjj     |   v1.0    | 2021/10/19 | split data & generate .txt
# ------------------------------------------------------------------------
'''

import os
import sys
sys.path.append('/media/hp3090/HDD-2T/renjunjie/WSOL_RS/')

import random
from tqdm import tqdm

random.seed(8)

train_txt = open('../dataset/PatternNetV2/PatternNetV2_train.txt', 'w')
val_txt = open('../dataset/PatternNetV2/PatternNetV2_test.txt', 'w')
label_txt = open('../dataset/PatternNetV2/PatternNetV2.txt', 'w')

train_ratio = 0.8

data_root = "/media/hp3090/HDD-2T/renjunjie/WSOL_RS/dataset/PatternNetV2/Images/"
label_list = []
for dir in tqdm(os.listdir(data_root)):
    print(dir)
    if dir not in label_list:
        label_list.append(dir)
        label_txt.write('{} {}\n'.format(dir, str(len(label_list)-1)))
        data_path = os.path.join(data_root, dir)
        train_list = random.sample(os.listdir(data_path), 
                                   int(len(os.listdir(data_path))*train_ratio))
        for im in train_list:
            train_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list)-1)))
        for im in os.listdir(data_path):
            if im in train_list:
                continue
            else:
                val_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list)-1)))

train_txt.close()
val_txt.close()
label_txt.close()


