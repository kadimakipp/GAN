#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: cgan.py
@time: 2019/9/9 下午6:35
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision

from net.cgan import *
from dataset.mnist import Mnist
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 12.0)
#plt.rcParams['figure.dpi'] = 300

class CGAN_Net(object):
    def __init__(self,latent_dim, image_shape, auxiliary,lr,b1,b2):

        self.latent_dim = latent_dim
        self.auxiliary_size = auxiliary
        self.generator = Generator(latent_dim, image_shape,auxiliary)
        self.discriminator = Discriminator(image_shape, auxiliary)

        self.adversarial_loss = torch.nn.BCELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr,betas=(b1,b2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(),lr=lr,betas=(b1,b2))

        data = Mnist()
        self.dataloader = data.get_loader(True, 512,28)

    def Train(self, epochs):
        plt.figure()
        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):
                valid = torch.ones((imgs.size(0),1)).to(self.device)
                fake = torch.zeros((imgs.size(0),1)).to(self.device)
                real_imgs = imgs.to(self.device)
                auxiliary = labels.to(self.device).type(torch.cuda.LongTensor)

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                np_z = np.random.normal(0,1, (imgs.shape[0], self.latent_dim))
                z = torch.from_numpy(np_z)
                z = z.to(self.device).type(torch.cuda.FloatTensor)

                np_aux = np.random.randint(0,self.auxiliary_size,imgs.shape[0])
                gen_aux = torch.from_numpy(np_aux)
                gen_aux =gen_aux.to(self.device).type(torch.cuda.LongTensor)


                gen_imgs = self.generator(z,gen_aux)
                valid_imgs = self.discriminator(gen_imgs,gen_aux)
                g_loss = self.adversarial_loss(valid_imgs,valid)
                g_loss.backward()
                self.optimizer_G.step()
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                real_dis = self.discriminator(real_imgs, auxiliary)
                real_loss = self.adversarial_loss(real_dis, valid)

                fake_dis = self.discriminator(gen_imgs.detach(), gen_aux)
                fake_loss = self.adversarial_loss(fake_dis, fake)
                d_loss = (real_loss + fake_loss)/2
                d_loss.backward()
                self.optimizer_D.step()

                if (i+1)%(len(self.dataloader)//2)==0:
                    print(
                        "[Epoch %3d/%3d] [Batch %3d/%3d] [real loss: %.4f] [fake loss %.4f] [G loss %.4f]"
                        % (epoch, epochs, i, len(self.dataloader), real_loss.item(), fake_loss.item(), g_loss.item())
                    )
                    self.sample_images(10)

    def sample_images(self, n_row):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        np_z = np.random.normal(0, 1, (n_row**2, self.latent_dim)).astype(np.float32)
        z = torch.from_numpy(np_z)
        z = z.to(self.device)
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        auxiliary = torch.from_numpy(labels)
        auxiliary = auxiliary.to(self.device).type(torch.cuda.LongTensor)
        gen_imgs = self.generator(z, auxiliary)
        dis_imgs = gen_imgs.view(n_row**2, *(28, 28))
        dis_imgs = dis_imgs.to("cpu")
        dis_imgs = dis_imgs.detach().numpy()
        for k, dis_img in enumerate(dis_imgs):
            plt.subplot(n_row, n_row, k + 1)
            plt.imshow(dis_imgs[k])
        plt.pause(1)


def main(epochs=200,latent_dim=100, auxiliary=10,image_shape=(1,28,28), lr=0.0005, b1=0.5, b2=0.999):
    cgan = CGAN_Net(latent_dim, image_shape,auxiliary,lr,b1,b2)
    cgan.Train(epochs)

if __name__ == "__main__":
    import fire

    fire.Fire(main)