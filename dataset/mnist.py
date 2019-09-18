#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: mnist.py
@time: 2019/9/4 下午6:33
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class Mnist(object):
    def __init__(self):
        self.root = "/media/kipp/work/pytorch_data"
        #self.root = "/media/kipp/work/Datas/MNIST"
        self.num_workers = 4
        self.shuffle=True

    def image_transforms(self,image_size):
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        )

        return transform


    def get_loader(self, train, batch_size, image_size ,download=True):
        trfm = self.image_transforms(image_size)
        dataset = torchvision.datasets.MNIST(self.root, train,trfm,download=download)
        return torch.utils.data.DataLoader(
            dataset,batch_size,self.shuffle,num_workers=self.num_workers
        )





def main():
    mnist = Mnist()
    it = mnist.get_loader(True, 1, 28)
    for i , (images, _) in enumerate(it):
        print(i, images.shape)
        if i>10:
            break




if __name__ == "__main__":
    import fire

    fire.Fire(main)