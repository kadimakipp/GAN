#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: sgan.py
@time: 2019/9/18 下午8:27
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
from net.sgan import *
from dataset.mnist import  Mnist
import matplotlib.pyplot as plt

class SGAN_Net(object):
    def __init__(self, latent_dim, num_classes, channels, img_size, lr, b1, b2):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.channels = channels
        self.img_size = img_size

        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()

        self.generator = Generator(self.latent_dim,self.num_classes, self.img_size,self.channels)
        self.discriminator = Discriminator(self.num_classes, self.channels, self.img_size)

        self.device = AuxF.device()

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.generator.apply(AuxF.weights_init_normal)
        self.discriminator.apply(AuxF.weights_init_normal)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),lr=lr, betas=(b1,b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1,b2))
        mnist = Mnist()
        self.loader = mnist.get_loader(True, 64, 32)

    def Train(self, epochs):
        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(self.loader):
                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)
                fake_aux_gt = torch.ones(batch_size).fill_(self.num_classes).type(torch.LongTensor)
                fake_aux_gt = fake_aux_gt.to(self.device)

                #Configure input
                real_imgs = imgs.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                np_z = np.random.normal(0, 1, (batch_size, self.latent_dim)).astype(np.float32)
                z = torch.from_numpy(np_z)
                z = z.to(self.device)
                gen_imgs = self.generator(z)

                validity, _ = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()

                self.optimizer_G.step()

                # --------------
                # Train Discriminator
                #---------------
                self.optimizer_D.zero_grad()

                real_pred, real_aux = self.discriminator(real_imgs)
                d_real_loss = (self.adversarial_loss(real_pred, valid)+self.auxiliary_loss(real_aux, labels))/2

                # Loss for fake images
                fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
                d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, fake_aux_gt)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                self.optimizer_D.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch, epochs, i, len(self.loader), d_loss.item(), 100 * d_acc, g_loss.item())
                )
                if (i + 1) % (len(self.loader) // 2) == 0:
                    dis_imgs = gen_imgs.data[:25].cpu()
                    dis_imgs = dis_imgs.view(25, *(self.img_size, self.img_size)).numpy()
                    for k, dis_img in enumerate(dis_imgs):
                        plt.subplot(5, 5, k + 1)
                        plt.imshow(dis_imgs[k])
                    plt.pause(1)


def main(epochs=200, latent_dim=100, num_classes=10, channels=1, img_size=32, lr=0.0002, b1=0.5, b2=0.999):
    sgan = SGAN_Net(latent_dim, num_classes, channels, img_size, lr, b1, b2)
    sgan.Train(epochs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)