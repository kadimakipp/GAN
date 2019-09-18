#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: sgan.py
@time: 2019/9/18 下午7:59
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels

        self.init_size = self.img_size//4

        self.l1 = nn.Sequential(
            nn.Linear(self.latent_dim, 128*self.init_size**2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3, stride=1, padding=1),
            nn.BatchNorm2d(128,0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes,channels, img_size):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        def discriminator_block(in_filters, out_filters, bn=True):
            """Return layers of each discriminator block"""

            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=True),
            *discriminator_block(16,32),
            *discriminator_block(32,64),
            *discriminator_block(64, 128),
        )

        # The height and width of down-sampled image
        ds_size = img_size//2**4

        #Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128*ds_size**2, self.num_classes+1), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label



