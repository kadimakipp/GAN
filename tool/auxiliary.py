#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: auxiliary.py
@time: 2019/9/18 下午8:00
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision

class AuxFunction(object):
    def __init__(self):
        pass

    @staticmethod
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") !=-1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    @staticmethod
    def device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")