'''
Author: TuZhou
Version: 1.0
Date: 2021-08-18 16:02:06
LastEditTime: 2021-08-18 16:28:15
LastEditors: TuZhou
Description: 
FilePath: \my_yolov5\nets\CSPDarknet53.py
'''

import torch
import torch.nn as nn


#----------------------------------------------------------------#
#   CBL模块：conv + batchNormalization + leakyRelu(或者为silu激活)
#----------------------------------------------------------------#
class CBL(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

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
        self.conv = CBL(self, in_channels, out_channels, kernel_size=1, stride=1, pad=None, groups=1, act=True)

    #切片步长为2，切片后图片尺寸减半，但是通道数变4倍，即12
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))



#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块,经过残差块的张量通道数不会改变
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            CBL(channels, hidden_channels, 1),
            CBL(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)