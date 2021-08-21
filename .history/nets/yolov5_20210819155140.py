'''
Author: TuZhou
Version: 1.0
Date: 2021-08-18 15:24:52
LastEditTime: 2021-08-19 15:51:40
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

    nc = yaml_dict['nc']

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
            
    return nc, blocks, filters


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
        self.in_channel = 512
        self.nc, self.number_blocks, self.filters_info = process_yaml()
        self.result_channels = 3 * (self.nc + 5)

        #---------------------------------------------------#
        #   第一个head特征融合处理
        #---------------------------------------------------#
        #filter:256x512x1x1,stride = 1
        self.conv1 = CBL(self.in_channel, self.filters_info[0][0], self.filters_info[0][1], self.filters_info[0][2])
        self.upsample1 = nn.Upsample(scale_factor=self.filter_info[1][1], mode=self.filters_info[1][2])
        #self.cat拼接
        #第一个csp2是在拼接后，所以输入通道维度翻倍了
        self.block1 = CSP2(self.filter_info[0][1]*2, self.filter_info[2][0], self.number_blocks[0])
        self.conv2 = CBL(self.filter_info[2][0], self.filter_info[3][0], self.filter_info[3][1], self.filter_info[3][2])
        self.upsample2 = nn.Upsample(scale_factor=self.filter_info[4][1], mode=self.filters_info[4][2])
        #self.cat拼接
        self.block2 = CSP2(self.filter_info[3][0]*2, self.filter_info[5][0], self.number_blocks[1])

        #---------------------------------------------------#
        #   第二个head特征融合处理
        #---------------------------------------------------#
        self.conv3 = CBL(self.filter_info[5][0], self.filter_info[6][0], self.filter_info[6][1], self.filter_info[6][2])
        #self.cat拼接
        self.block3 = CSP2(self.filter_info[6][0]*2, self.filter_info[7][0], self.filter_info[7][1], self.number_blocks[2])

        #---------------------------------------------------#
        #   第三个head特征融合处理
        #---------------------------------------------------#
        self.conv4 = CBL(self.filter_info[7][0], self.filter_info[8][0], self.filter_info[8][1], self.filter_info[8][2])
        #self.cat拼接
        self.block4 = CSP2(self.filter_info[8][0]*2, self.filter_info[9][0], self.filter_info[9][1], self.number_blocks[3])


    def forward(self, x):
        #backbone, 从2到0特征图尺寸依次从大到小
        x2, x1, x0 = self.backbone(x)

        #第一个76x76特征图融合处理
        y0 = self.conv1(x0)                 #待融合
        x0 = torch.cat([self.upsample1(y0), x1], 1)
        y1 = self.conv2(self.block1(x0))        #待融合
        out0 = self.block2(torch.cat([self.upsample2(y1), x2], 1))

        #第二个38x38特征图融合处理
        out1 = self.block3(torch.cat([self.conv3(out0), y1], 1))

        #第三个19x19特征图融合处理
        out3 = self.block4(torch.cat([self.conv4(out1), y0], 1))

        #经过head部分装换结果输出通道维度
        out0 = nn.conv2(self.filter_info[5][0], self.result_channels, 1, 1)
        out1 = nn.conv2(self.filter_info[7][0], self.result_channels, 1, 1)
        out2 = nn.conv2(self.filter_info[9][0], self.result_channels, 1, 1)

        #输出特征图尺寸从小到大排列，由19x19到76x76
        return out2, out1, out0
