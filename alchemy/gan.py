#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: gan.py
@time: 2019/9/4 下午7:39
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision

from net.gan import *
from dataset.mnist import Mnist
import matplotlib.pyplot as plt

class GAN_Net(object):
    def __init__(self, latent_dim, image_shape, lr, b1, b2):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, image_shape)
        self.discriminator = Discriminator(image_shape)
        self.adversarial_loss = torch.nn.BCELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(b1,b2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(),lr=lr,betas=(b1,b2))

        data = Mnist()
        self.dataloader = data.get_loader(True, 512, 28)

    def Train(self, epochs):
        plt.figure()
        for epoch in range(epochs):
            for i, (imgs,_) in enumerate(self.dataloader):
                #GT
                valid = torch.ones((imgs.size(0),1)).to(self.device)
                fake = torch.zeros((imgs.size(0),1)).to(self.device)
                real_imgs = imgs.to(self.device)
                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                np_z = np.random.normal(0,1,(imgs.shape[0],self.latent_dim)).astype(np.float32)
                z = torch.from_numpy(np_z)
                z = z.to(self.device)

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

                if (i+1)%(len(self.dataloader)//6)==0:
                    print(
                        "[Epoch %3d/%3d] [Batch %3d/%3d] [real loss: %.4f] [fake loss %.4f] [G loss %.4f]"
                        % (epoch, epochs, i, len(self.dataloader), real_loss.item(), fake_loss.item(), g_loss.item())
                    )
                    dis_imgs = gen_imgs.data[:25].to("cpu")
                    dis_imgs = dis_imgs.view(25, *(28,28)).numpy()
                    for k, dis_img in enumerate(dis_imgs):
                            plt.subplot(5,5, k+1)
                            plt.imshow(dis_imgs[k])
                    plt.pause(1)

def main(epochs=200,latent_dim=100, image_shape=(1,28,28), lr=0.0005, b1=0.5, b2=0.999):
    gan = GAN_Net(latent_dim, image_shape,lr,b1,b2)
    gan.Train(epochs)

if __name__ == "__main__":
    import fire
    fire.Fire(main)