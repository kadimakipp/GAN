#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: mini_imagenet.py
@time: 2019/9/17 下午6:52
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
from torch.utils.data import Dataset
from PIL import Image
import glob

class ImageDataset(Dataset):
    def __init__(self, root, transform_=None, image_size=128, mask_size=64,mode="train"):
        self.transform = transforms.Compose(transform_)
        self.img_size = image_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg"%root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]


    def apply_random_mask(self, img):
        "Randomly masks image"
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        "Mask cneter part of images"
        #Get upper-left pixel coordinate
        i=(self.img_size - self.mask_size)//2
        masked_img = img.clone()
        masked_img[:, i:i+self.mask_size, i : i+self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index%len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)



class CelebA(object):
    def __init__(self):
        self.root = "/media/kipp/work/Datas/celeba-dataset/img_align_celeba"
        self.work_num = 4
        self.shuffle = True

    def transform_(self, img_size):
        transform = [
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
        return transform

    def get_loader(self, batch_size, img_size=128, mask_size=64,mode="train",num_workers=4):
        transform = self.transform_(img_size)
        return torch.utils.data.DataLoader(
            ImageDataset(self.root, transform,img_size,mask_size,mode),
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers
        )

def main():
    celeba = CelebA()
    loader = celeba.get_loader(1, mode="train")
    for i, (img, mask, aux) in enumerate(loader):
        print(img.shape, mask.shape, aux.shape)
        if i >10:
            break
    loader = celeba.get_loader(1, mode="val")
    for i, (img, mask, aux) in enumerate(loader):
        print(img.shape, mask.shape, aux)
        if i > 10:
            break


if __name__ == "__main__":
    import fire
    fire.Fire(main)