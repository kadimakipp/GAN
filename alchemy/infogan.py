#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: infogan.py
@time: 2019/9/19 下午7:22
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision

from dataset.mnist import Mnist
from net.infogan import *
from tool.auxiliary import AuxFunction as AuxF
from torchvision.utils import save_image

import itertools
import os
os.makedirs("../images/infogan/static/", exist_ok=True)
os.makedirs("../images/infogan/varying_c1/", exist_ok=True)
os.makedirs("../images/infogan/varying_c2/", exist_ok=True)

class InfoGAN_Net(object):
    def __init__(self, batch_size, latent_dim, code_dim, n_classes, img_size, channels,lr,b1,b2):
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.n_classes = n_classes
        self.channels = channels
        self.img_size = img_size

        self.adversarial_loss = torch.nn.MSELoss()
        self.categorical_loss = torch.nn.CrossEntropyLoss()
        self.continuous_loss = torch.nn.MSELoss()

        self.lambda_cat = 1
        self.lambda_con = 0.1

        self.device = AuxF.device()

        self.generator = Generator(self.latent_dim,self.n_classes,
                                   self.code_dim, self.img_size,
                                   self.channels).to(self.device)
        self.discriminator = Discriminator(self.n_classes,self.code_dim,
                                           self.channels,self.img_size).to(self.device)
        self.generator.apply(AuxF.weights_init_normal)
        self.discriminator.apply(AuxF.weights_init_normal)

        mnist = Mnist()
        self.dataloader = mnist.get_loader("train", batch_size, img_size)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1,b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr = lr, betas=(b1,b2))
        self.optimizer_info = torch.optim.Adam(
            itertools.chain(self.generator.parameters(), self.discriminator.parameters()),
            lr=lr, betas=(b1,b2)
        )

        #Static generator inputs for sampleing
        self.__static_z = torch.zeros((self.n_classes ** 2, self.latent_dim)).to(self.device)
        y = np.array([num for _ in range(self.n_classes) for num in range(self.n_classes)])
        self.__static_label = AuxF.to_categorical(y, self.n_classes)
        self.__static_code = torch.zeros((self.n_classes ** 2, self.code_dim)).to(self.device)

    def sample_img(self,n_row, index):
        # Static sample
        z_size  = (n_row**2, self.latent_dim)
        z = torch.normal(torch.zeros(z_size), torch.ones(z_size)).to(self.device)
        static_sample = self.generator(z, self.__static_label, self.__static_code)
        save_image(static_sample.data, "../images/infogan/static/%d.png"%index, nrow=n_row, normalize=True)
        # Get varied c1 and c2
        zeros = torch.zeros((n_row**2, 1))
        c_varied = torch.repeat_interleave(torch.linspace(-1,1,n_row).unsqueeze_(1), n_row, 0)
        c1 = torch.cat((c_varied, zeros), -1).to(self.device)
        c2 = torch.cat((zeros, c_varied), -1).to(self.device)
        sample1 = self.generator(self.__static_z, self.__static_label, c1)
        sample2 = self.generator(self.__static_z, self.__static_label, c2)
        save_image(sample1.data, "../images/infogan/varying_c1/%d.png" % index, nrow=n_row, normalize=True)
        save_image(sample2.data, "../images/infogan/varying_c2/%d.png" % index, nrow=n_row, normalize=True)


    def Train(self, epochs):
        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):
                batch_size = imgs.shape[0]

                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)

                real_img = imgs.to(self.device)
                labels = AuxF.to_categorical(labels, num_columns=self.n_classes)
                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                z = torch.normal(torch.zeros((batch_size, self.latent_dim)), torch.ones((batch_size, self.latent_dim)))
                z= z.to(self.device)
                label_c = torch.randint(0, self.n_classes, (batch_size,))
                label_input = AuxF.to_categorical(label_c, num_columns=self.n_classes)
                code_input = torch.Tensor(batch_size, self.code_dim).uniform_(-1,1).to(self.device)
                #Generate a batch of images
                gen_imgs = self.generator(z, label_input, code_input)
                validity,_,_ = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                # Loss for real images
                real_pred, _,_ = self.discriminator(real_img)
                d_real_loss = self.adversarial_loss(real_pred, valid)

                # Loss for fake images
                fake_pred,_,_ = self.discriminator(gen_imgs.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake)
                d_loss = (d_real_loss + d_fake_loss)/2

                d_loss.backward()
                self.optimizer_D.step()

                # ------------------
                # Information Loss
                # ------------------
                self.optimizer_info.zero_grad()
                # Sample labels
                #sampled_labels = np.random.randint(0, self.n_classes, batch_size)
                gt_labels = torch.randint(0, self.n_classes, (batch_size,),dtype=torch.long,requires_grad=False,device=self.device)

                z = torch.normal(torch.zeros((batch_size, self.latent_dim)), torch.ones((batch_size, self.latent_dim)))
                z = z.to(self.device)

                label_input = AuxF.to_categorical(gt_labels, self.n_classes)

                code_input = torch.Tensor(batch_size, self.code_dim).uniform_(-1,1).to(self.device)
                gen_imgs = self.generator(z, label_input, code_input)
                _, pred_label, pred_code = self.discriminator(gen_imgs)

                info_loss = self.lambda_cat * self.categorical_loss(pred_label, gt_labels)\
                            + self.lambda_con * self.continuous_loss(pred_code, code_input)

                info_loss.backward()
                self.optimizer_info.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                    % (epoch, epochs, i, len(self.dataloader), d_loss.item(), g_loss.item(), info_loss.item())
                )

                if (i + 1) % (len(self.dataloader) // 2) == 0:
                    batches_done = epoch * len(self.dataloader) + i
                    self.sample_img(10,batches_done)

def main(epochs=200, batch_size=64,
         latent_dim=62, code_dim=2, n_classes=10,
         img_size=32, channels=1,lr=0.0002,b1=0.5,b2=0.99):
    infogan = InfoGAN_Net(batch_size, latent_dim, code_dim, n_classes, img_size, channels,lr,b1,b2)
    infogan.Train(epochs)



if __name__ == "__main__":
    import fire

    fire.Fire(main)