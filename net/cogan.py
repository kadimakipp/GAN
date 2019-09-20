#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: cogan.py
@time: 2019/9/20 下午7:28
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

"""Coupled GAN"""

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = img_size//4
        self.channels = channels

        self.fc = nn.Sequential(nn.Linear(self.latent_dim,128*self.init_size**2))

        self.shared_conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
        )
        self.G1 = nn.Sequential(
            nn.Conv2d(128,64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.G2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,self.channels, 3, stride=1,padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)

        return img1, img2

class Discriminators(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminators, self).__init__()

        self.ds_size = img_size//2 **4
        self.channels = channels

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3,2,1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True),
                          nn.Dropout2d(0.25)])
            return block

        self.shared_conv = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16,32),
            *discriminator_block(32,64),
            *discriminator_block(64,128),
        )
        self.D1 = nn.Linear(128*self.ds_size**2, 1)
        self.D2 = nn.Linear(128*self.ds_size**2, 1)

    def forward(self, img1, img2):
        # Determine validity of first image
        out = self.shared_conv(img1)
        out = out.view(out.shape[0],-1)
        validity1 = self.D1(out)
        out = self.shared_conv(img2)
        out = out.view(out.shape[0],-1)
        validity2 = self.D2(out)

        return validity1, validity2




def main():
    pass


if __name__ == "__main__":
    import fire

    fire.Fire(main)