#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: tiny_imagenet.py
@time: 2019/9/23 下午6:24
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Tiny Imagenet has 200 classes. 
Each class has 500 training images, 50 validation images, and 50 test images.
We have released the training and validation sets with images and annotations. 
We provide both class labels and bounding boxes as annotations;
however, you are asked only to predict the class label of each image without localizing the objects.
"""
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
from PIL import Image
import os
import pandas as pd

class TinyImagenet(Dataset):
    def __init__(self,root, transform, train="train"):
        super(TinyImagenet,self).__init__()
        self.transform = transform

        if train not in ["train","val", "test"]:
            RuntimeError("train must be in ('train, val, test'),but train = %s"%(train))
        self.root = os.path.join(root, train)
        columns = ["filename", "wnid", "b1", "b2", "b3", "b4"]
        self.data = pd.DataFrame(columns=columns)
        if train == "train":
            self.wnids = open(os.path.join(root, "wnids.txt"), "r").readlines()
            self.wnids = map(lambda s: s.replace('\n', ''), self.wnids)
            self.classes_dict = {}
            for i, wnid in enumerate(self.wnids):
                self.classes_dict.update({wnid:i})
                txt = os.path.join(wnid, wnid+"_boxes.txt")
                boxes = open(os.path.join(self.root, txt), 'r').readlines()
                for box in boxes:
                    box = box.replace('\n', '')
                    filename, b1,b2,b3,b4 = box.split('\t')
                    d = pd.DataFrame([[filename, wnid,
                                       float(b1), float(b2),
                                       float(b3), float(b4)]],columns=columns)
                    self.data = self.data.append(d, ignore_index=True)
                if i>2:
                    break
        self.data = self.data.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        filename, wnid, b1,b2,b3,b4 = self.data[item]
        filename = os.path.join("images", filename)
        img_name = os.path.join(wnid, filename)
        image = Image.open(os.path.join(self.root,img_name))
        label = self.classes_dict[wnid]
        bbox = [b1,b2,b3,b4]
        return image, label, bbox

class tinyImagenet(object):
    def __init__(self):
        self.root = "/media/kipp/work/Datas/tiny-imagenet-200"
        self.num_work=4
        self.shuffle=True

    def Transform(self, img_size):
        transform = [
            # Transforms.RandomCrop(224),
            # Transforms.RandomHorizontalFlip(0.5),
            # Transforms.RandomAffine(5),
            Transforms.Resize((img_size, img_size), Image.BICUBIC),
            Transforms.ToTensor(),
            Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        return transform

    def get_loader(self, batch_size, img_size, mode="train"):
        transform = self.Transform(img_size)
        return torch.utils.data.DataLoader(
            TinyImagenet(self.root,transform, train=mode),
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers = self.num_work
        )

def main():
    tiny = tinyImagenet()
    loader = tiny.get_loader(1,64)
    for i, (image, label, box) in enumerate(loader):
        if i>2:
            break
        print(image.shape, label, box)





if __name__ == "__main__":
    import fire

    fire.Fire(main)