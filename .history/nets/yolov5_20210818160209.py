'''
Author: TuZhou
Version: 1.0
Date: 2021-08-18 15:24:52
LastEditTime: 2021-08-18 16:02:09
LastEditors: TuZhou
Description: 
FilePath: \my_yolov5\nets\yolov5.py
'''


import torch
import torch.nn as nn


class YoloModel(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict