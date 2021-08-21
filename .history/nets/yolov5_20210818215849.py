'''
Author: TuZhou
Version: 1.0
Date: 2021-08-18 15:24:52
LastEditTime: 2021-08-18 21:58:49
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
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict