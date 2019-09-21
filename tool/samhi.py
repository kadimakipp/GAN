#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: samhi.py
@time: 2019/9/20 下午7:03
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
from tool.auxiliary import AuxFunction as AuxF
from dataset.mnistm import MnistM
from dataset.mnist import Mnist
#from abc import ABCMeta, abstractmethod #meta class
class SAMHI(object):
    def __init__(self, batch_size,latent_dim,img_size, channels, lr, b1, b2):
        self._batch_size = batch_size
        self._latent_dim =latent_dim
        self._img_size = img_size
        self._channels = channels
        self._lr = lr
        self._b1 = b1
        self._b2 = b2

        self._device =  AuxF.device()


    def optimizer(self, generator, discriminator):
        self._optimizer_G = torch.optim.Adam(generator.parameters(), lr=self._lr, betas=(self._b1, self._b2))
        self._optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=self._lr, betas=(self._b1, self._b2))


    def weights_init(self, net):
        net.apply(AuxF.weights_init_normal)

    def get_mnist_loader(self,train):
        mnist = Mnist()
        mnist_loader = mnist.get_loader(train, self._batch_size, self._img_size)
        return mnist_loader

    def get_mnistm_loader(self, train):
        mnistm = MnistM()
        mnistm_loader = mnistm.get_loader(batch_size=self._batch_size,
                                          img_size=self._img_size,
                                          train=train)
        return mnistm_loader
