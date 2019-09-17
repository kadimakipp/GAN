#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: dcgan.py
@time: 2019/9/11 下午6:35
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
from net.dcgan import *
from dataset.mnist import Mnist
import matplotlib.pyplot as plt

class DCGAN_Net(object):
    def __init__(self, latent_dim, img_shape,lr, b1, b2):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.adversarial_loss = torch.nn.BCELoss()
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)
        self.generator.apply(self.weights_init_normal)
        self.discriminator.apply(self.weights_init_normal)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),lr=lr,betas=(b1,b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1,b2))

        data = Mnist()
        self.dataloader = data.get_loader(True, 512, 32)

    @staticmethod
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv")!= -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") !=-1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def Train(self, epochs):
        for epoch in range(epochs):
            for i, (imgs,_) in enumerate(self.dataloader):
                valid = torch.ones((imgs.size(0), 1)).to(self.device)
                fake = torch.zeros((imgs.size(0), 1)).to(self.device)
                real_imgs = imgs.to(self.device)
                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                z = torch.from_numpy(np.random.normal(0,1,(imgs.shape[0], self.latent_dim)))
                z = z.type(torch.FloatTensor).to(self.device)
                gen_imgs = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                real_loss = self.adversarial_loss(self.discriminator(real_imgs),valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()),fake)
                d_loss = (real_loss+fake_loss)/2
                d_loss.backward()
                self.optimizer_D.step()

                if (i + 1) % (len(self.dataloader) // 2) == 0:
                    print(
                        "[Epoch %3d/%3d] [Batch %3d/%3d] [real loss: %.4f] [fake loss %.4f] [G loss %.4f]"
                        % (epoch, epochs, i, len(self.dataloader), real_loss.item(), fake_loss.item(), g_loss.item())
                    )
                    dis_imgs = gen_imgs.data[:25].to("cpu")
                    dis_imgs = dis_imgs.view(25, *(self.img_shape[1], self.img_shape[1])).numpy()
                    for k, dis_img in enumerate(dis_imgs):
                        plt.subplot(5, 5, k + 1)
                        plt.imshow(dis_imgs[k])
                    plt.pause(1)

def main(epochs=200,latent_dim=100, image_shape=(1,32,32), lr=0.0002, b1=0.5, b2=0.999):
    dcgan = DCGAN_Net(latent_dim,image_shape,lr,b1,b2)
    dcgan.Train(epochs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)