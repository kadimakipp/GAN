#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: aae.py
@time: 2019/9/10 下午6:59
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import itertools
from net.aae import *
from dataset.mnist import Mnist
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 12.0)

class AAE_Net(object):
    def __init__(self, latent_dim, img_shape,lr,b1,b2):
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.adversarial_loss  = nn.BCELoss()
        self.pixewise_loss = nn.L1Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(self.latent_dim, self.img_shape).to(self.device)
        self.decoder = Decoder(self.latent_dim, self.img_shape).to(self.device)
        self.discriminator = Discriminator(self.latent_dim).to(self.device)

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(),self.decoder.parameters()),lr=lr,betas=(b1,b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),lr=lr,betas=(b1,b2))

        data = Mnist()
        self.dataloader = data.get_loader(True, 512, 28)

    def sample_images(self, n_row):
        z = torch.from_numpy(np.random.normal(0,1,(n_row**2,self.latent_dim))).type(torch.cuda.FloatTensor)
        z = z.to(self.device)
        gen_imgs = self.decoder(z)
        dis_imgs = gen_imgs.view(n_row ** 2, *(28, 28))
        dis_imgs = dis_imgs.to("cpu")
        dis_imgs = dis_imgs.detach().numpy()
        for k, dis_img in enumerate(dis_imgs):
            plt.subplot(n_row, n_row, k + 1)
            plt.imshow(dis_imgs[k])
        plt.pause(1)

    def Train(self, epochs):
        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):
                valid = torch.ones((imgs.size(0), 1)).to(self.device)
                fake = torch.zeros((imgs.size(0), 1)).to(self.device)
                real_imgs = imgs.to(self.device)
                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)
                coder_loss = 0.001*self.adversarial_loss(self.discriminator(encoded_imgs), valid)
                g_loss = coder_loss+ 0.999*self.pixewise_loss(decoded_imgs, real_imgs)
                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                z = torch.from_numpy(np.random.normal(0,1,(imgs.shape[0], self.latent_dim)))
                z = z.to(self.device).type(torch.cuda.FloatTensor)
                real_loss = self.adversarial_loss(self.discriminator(z),valid)
                fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach()),fake)
                d_loss = 0.5*(real_loss+fake_loss)

                d_loss.backward()
                self.optimizer_D.step()
                if(i + 1) % (len(self.dataloader) // 2) == 0:
                    print(
                        "[Epoch %3d/%3d] [Batch %3d/%3d] [real loss: %.4f] [fake loss %.4f] [G loss %.4f]"
                        % (epoch, epochs, i, len(self.dataloader), real_loss.item(), fake_loss.item(), g_loss.item())
                    )
                    self.sample_images(10)


def main(epochs=250,latent_dim=100, image_shape=(1,28,28), lr=0.0005, b1=0.5, b2=0.999):
    aae = AAE_Net(latent_dim, image_shape,lr,b1,b2)
    aae.Train(epochs)

if __name__ == "__main__":
    import fire

    fire.Fire(main)