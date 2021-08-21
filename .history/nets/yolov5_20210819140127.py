'''
Author: TuZhou
Version: 1.0
Date: 2021-08-18 15:24:52
LastEditTime: 2021-08-19 14:01:27
LastEditors: TuZhou
Description: 
FilePath: \my_yolov5\nets\yolov5.py
'''


import torch
import torch.nn as nn
from pathlib import Path
import yaml
import math

from CSPDarknet53 import preprocess_backbone, CSP2, CBL


#------------------------------#
#   计算输出通道数
#------------------------------#
def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


#------------------------------#
#   处理yaml文件
#   读取出backbone模块的信息
#------------------------------#
def process_yaml(yaml_path = './nets/yolov5s.yaml'):
    yaml_file = Path(yaml_path)
    with open(yaml_file, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    #提取出网络宽度与深度
    gd = yaml_dict['depth_multiple']
    gw = yaml_dict['width_multiple']

    backbone_dict = yaml_dict['head']

    filters = []
    blocks = []

    for n in backbone_dict:
        if n[2] == 'C3':
            blocks.append(n[1])
        if not n[2] == 'Concat':
            filters.append(n[3])


    blocks = blocks[:]
    for i, _ in enumerate(blocks):
        blocks[i] = max(round(blocks[i] * gd), 1) if blocks[i] > 1 else blocks[i]

    for i, _ in enumerate(filters):
        if not isinstance(filters[i][0], str):
            filters[i][0] = make_divisible(filters[i][0] * gw, 8)
            
    return blocks, filters


class YoloBody(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', image_channels=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   以输入为608x608的图片为例
        #   获得三个有效特征层，他们的shape分别是：
        #   76,76,128
        #   38,38,256
        #   19,19,512
        #---------------------------------------------------#
        self.backbone = preprocess_backbone(None)
        self.number_blocks, self.filters_info = process_yaml()
        
        
        