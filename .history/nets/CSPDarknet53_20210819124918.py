'''
Author: TuZhou
Version: 1.0
Date: 2021-08-18 16:02:06
LastEditTime: 2021-08-19 12:49:18
LastEditors: TuZhou
Description: 
FilePath: \my_yolov5\nets\CSPDarknet53.py
'''

import torch
import torch.nn as nn

from pathlib import Path
import yaml
import math


#----------------------------------------------------------------#
#   CBL模块：conv + batchNormalization + leakyRelu(或者为silu激活)
#----------------------------------------------------------------#
class CBL(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

    #无标准化
    def forward_fuse(self, x):
        return self.activation(self.conv(x))


#focus层,进行切片拼接操作
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad=None, groups=1, act=True): 
        super().__init__()
        self.conv = CBL(in_channels*4, out_channels, kernel_size=3, stride=1, pad=None, groups=1)

    #切片步长为2，切片后图片尺寸减半，但是通道数变4倍，即12
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        print(x)
        return self.conv(x)



#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块,经过残差块的张量通道数不会改变
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        #残差模块的卷积输入输出通道是一样的
        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            CBL(channels, hidden_channels, 1),
            CBL(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


#---------------------------------------------------#
#   CSP1模块
#   该模块只应用于backbone结构
#---------------------------------------------------#
class CSP1(nn.Module): 
# CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, number_block, groups=1, gw=0.5):  
        super(CSP1, self).__init__()
        #经过残差层的直连边
        c_ = int(out_channels * gw)     # hidden channels
        self.cv1 = CBL(in_channels, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.resblock = nn.Sequential(*[Resblock(c_, c_) for _ in range(number_block)])

        #只经过一次卷积的短接边，应为1x1卷积
        self.cv3 = nn.Conv2d(in_channels, c_, 1, 1, bias=False)
        
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.activation = nn.LeakyReLU(0.1, inplace=True)       #True表示改变输入的数据

        #最后一次CBL是1x1卷积，提升维度
        self.cv4 = CBL(2 * c_, out_channels, 1, 1)
    

    def forward(self, x):
        y1 = self.cv2(self.resblock(self.cv1(x)))          #经过残差层的直连边
        y2 = self.cv3(x)            #经过一次卷积的短接边
        return self.cv4(self.activation(self.bn(torch.cat((y1, y2), dim=1))))


#------------------------------#
#   CSP2模块，此模块不带有残差块
#------------------------------#
class CSP2(nn.Module): 
    def __init__(self, in_channels, out_channels, number_block, groups=1, gw=0.5):  
        super(CSP2, self).__init__()
        c_ = int(out_channels * gw)     # hidden channels

        #作为直连边
        self.conv1 = CBL(in_channels, c_, 1, 1)
        self.convBlock = nn.ModuleList([CBL(c_, c_, kernel_size = 1), CBL(c_, c_, kernel_size = 3)])
        self.conv2 = nn.Sequential(*[self.convBlock for _ in range(number_block)])
        self.conv3 = nn.Conv2d(c_, c_, 1, 1)

        #作为短接边
        self.conv4 = nn.Conv2d(in_channels, c_, 1, 1)

        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.activation = nn.LeakyReLU(0.1, inplace=True)       #True表示改变输入的数据
        self.conv5 = CBL(out_channels, out_channels, kernel_size = 1, stride = 1)

    def forward(self, x):
        #作为直连边
        y1 = self.conv3(self.conv2(self.conv1(x)))
        #作为短接边
        y2 = self.conv4(x)

        return self.cv5(self.activation(self.bn(torch.cat((y1, y2), dim=1))))



#---------------------------------------------------#
#   SPP模块
#   即做对输入做几种尺寸的最大池化再拼接，可获得固定输出的通道数
#---------------------------------------------------#
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = CBL(in_channels, c_, 1, 1)
        #拼接后输入的通道数变为原来的4倍
        self.cv2 = CBL(c_ * (len(k) + 1), out_channels, 1, 1)
        #padding = x//2可以使每种kernel_size的最大池化后图片尺寸不会变化
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)       
        #两个列表相加相当于extend，也就是扩展维度而不是add
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))



class CSPDarknet(nn.Module):
    def __init__(self, image_channel, number_blocks, filters_info):
        super().__init__()
        #print(filters_info[0][1])
        self.focus = Focus(image_channel, filters_info[0][0], kernel_size=filters_info[0][1],\
                                stride=1, pad=None, groups=1, act=True)

        #------------------------------#
        # 此处很简单，顺序就是卷->瓶颈->卷->瓶颈->卷->瓶颈->卷->spp->csp2
        #瓶颈就是csp1
        #------------------------------#

        #每次单独的CBL都是一次降采样，步长为2
        self.conv1 = CBL(filters_info[0][0], filters_info[1][0], kernel_size = filters_info[1][1], stride = filters_info[1][2])
        
        self.block1 = CSP1(filters_info[1][0], filters_info[2][0], number_block = number_blocks[0])      

        self.conv2 = CBL(filters_info[2][0], filters_info[3][0], kernel_size = filters_info[3][1], stride = filters_info[3][2])

        self.block2 = CSP1(filters_info[3][0], filters_info[4][0], number_block = number_blocks[1])

        self.conv3 = CBL(filters_info[4][0], filters_info[5][0], kernel_size = filters_info[5][1], stride = filters_info[5][2])

        self.block3 = CSP1(filters_info[5][0], filters_info[6][0], number_block = number_blocks[2])

        self.conv4 = CBL(filters_info[6][0], filters_info[7][0], kernel_size = filters_info[7][1], stride = filters_info[7][2])

        self.spp = SPP(filters_info[7][0], filters_info[8][0], filters_info[8][1])

        self.csp2 = CSP2(filters_info[8][0], filters_info[9][0], number_block = number_blocks[3])

    def forward(self, x):
        x = self.focus(x)
        print(0)
        x = self.conv1(x)
        print(1)
        x = self.conv2(self.blocks(x))
        print(2)
        #将作为backbone的输出之一，用于neck模块的融合
        out1 = self.block2(x)
        print(3)
        out2 = self.block3(self.conv3(out1))
        print(4)
        #print(out2)
        out3 = self.csp2(self.spp(self.conv4(out2)))

        return out1, out2, out3
        


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
    with open(yaml_file) as f:
        yaml_dict = yaml.safe_load(f)

    #提取出网络宽度与深度
    gd = yaml_dict['depth_multiple']
    gw = yaml_dict['width_multiple']

    backbone_dict = yaml_dict['backbone']
    
    filters = []
    blocks = []

    #计算CSP层块数与每个模块的filters等信息
    for n in backbone_dict:
        if n[2] == 'C3':
            blocks.append(n[1])
        filters.append(n[3])

    blocks = blocks[:]
    for i, _ in enumerate(blocks):
        blocks[i] = max(round(blocks[i] * gd), 1) if blocks[i] > 1 else blocks[i]

    for i, _ in enumerate(filters):
        filters[i][0] = make_divisible(filters[i][0] * gw, 8)
    print(blocks)
    print(filters)
    return blocks, filters


#------------------------------#
#   backbone模块的预处理
#   如读取预训练文件等
#------------------------------#
def preprocess_backbone(pretrained, **kwargs):
    number_blocks, filters_info = process_yaml()

    model = CSPDarknet(3, number_blocks, filters_info)
    
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == "__main__":
    model = preprocess_backbone(None)
    x = torch.rand(608, 608)
    torch.unsqueeze(x, 0)
    out1, out2, out3 = model(x)
    print(out1)
    #print(x)