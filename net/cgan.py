#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: cgan.py
@time: 2019/9/9 下午5:33
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
import torch.nn.functional as F
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape, auxiliary):
        super(Generator,self).__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.auxiliary_size = auxiliary
        self.label_emb = nn.Embedding(self.auxiliary_size,self.auxiliary_size)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.auxiliary_size+self.latent_dim, 128,normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh()
        )

    def forward(self, noise, auxiliary):
        gen_input = torch.cat((self.label_emb(auxiliary),noise),-1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.image_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, image_shape, auxiliary):
        super(Discriminator, self).__init__()

        self.image_shape = image_shape
        self.auxiliary_size = auxiliary
        self.label_emb = nn.Embedding(self.auxiliary_size,self.auxiliary_size)
        self.model = nn.Sequential(
            nn.Linear(self.auxiliary_size+int(np.prod(self.image_shape)),512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self, img, auxiliary):
        dis_input = torch.cat((self.label_emb(auxiliary),img.view(img.size(0),-1)),-1)
        validity = self.model(dis_input)
        return validity

