'''
Author: TuZhou
Version: 1.0
Date: 2021-08-18 15:24:52
LastEditTime: 2021-08-18 22:08:18
LastEditors: TuZhou
Description: 
FilePath: \my_yolov5\nets\yolov5.py
'''


import torch
import torch.nn as nn
from pathlib import Path
import yaml

from CSPDarknet53 import preprocess_backbone, CSP2


class YoloBody(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
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
        
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict