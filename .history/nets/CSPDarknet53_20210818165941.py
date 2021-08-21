'''
Author: TuZhou
Version: 1.0
Date: 2021-08-18 16:02:06
LastEditTime: 2021-08-18 16:59:41
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


#---------------------------------------------------#
#   CSP1模块
#   该模块只应用于backbone结构
#---------------------------------------------------#
class CSP1(nn.Module): 
# CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, number_block, groups=1, gw=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        #经过残差层的直连边
        c_ = int(out_channels * gw)     # hidden channels
        self.cv1 = CBL(in_channels, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.resblock = nn.Sequential(*[Resblock(c_, c_,  e=1.0) for _ in range(number_block)])

        #只经过一次卷积的短接边
        self.cv3 = nn.Conv2d(in_channels, c_, 1, 1, bias=False)
        
        self.cv4 = CBL(2 * c_, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.activation = nn.LeakyReLU(0.1, inplace=True)       #True表示改变输入的数据
        

    def forward(self, x):
        y1 = self.cv2(self.resblock(self.cv1(x)))          #经过残差层的直连边
        y2 = self.cv3(x)            #经过一次卷积的短接边
        return self.cv4(self.activation(self.bn(torch.cat((y1, y2), dim=1))))


#---------------------------------------------------#
#   SPP模块
#   即做对输入做几种尺寸的最大池化再拼接，可获得固定输出的通道数
#---------------------------------------------------#
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = CBL(in_channels, c_, 1, 1)
        self.cv2 = CBL(c_ * (len(k) + 1), out_channels, 1, 1)
        #padding = x//2可以使每种kernel_size的最大池化后图片尺寸不会变化
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
