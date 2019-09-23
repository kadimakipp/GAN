#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: cogan.py
@time: 2019/9/21 下午4:53
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import torch.nn as nn
from tool.samhi import SAMHI
from net.cogan import *
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 12.0)

"""
coupled GAN
"""
class COGAN_Net(SAMHI):
    def __init__(self, batch_size, latent_dim, img_size, channels, lr, b1,b2):
        super(COGAN_Net, self).__init__(batch_size=batch_size,
                                        latent_dim = latent_dim,
                                        img_size=img_size,
                                        channels=channels,
                                        lr=lr,b1=b1,b2=b2)
        self.adversarial_loss = nn.MSELoss()
        self.coupled_generators = Generator(self._latent_dim,
                                            self._img_size,
                                            self._channels).to(self._device)
        self.coupled_discriminators = Discriminators(self._img_size,
                                                     self._channels).to(self._device)

        self.weights_init(self.coupled_generators)
        self.weights_init(self.coupled_discriminators)

        self.mnist = self.get_mnist_loader(train=True)
        self.mnistm = self.get_mnistm_loader(train=True)

        self.optimizer(self.coupled_generators, self.coupled_discriminators)

    def Train(self, epochs):
        plt.figure()
        for epoch in range(epochs):
            for i, ((imgs1,_),(imgs2,_)) in enumerate(zip(self.mnist,self.mnistm)):
                batch_size = imgs1.shape[0]

                valid = torch.ones((batch_size,1)).to(self._device)
                fake = torch.zeros((batch_size,1)).to(self._device)

                #Configure input
                imgs1 = imgs1.expand(imgs1.shape[0], 3, self._img_size, self._img_size).to(self._device)
                imgs2 = imgs2.to(self._device)

                # ------------------
                #  Train Generators
                # ------------------
                self._optimizer_G.zero_grad()
                z = torch.normal(torch.zeros((batch_size,self._latent_dim)),
                                 torch.ones((batch_size,self._latent_dim))).to(self._device)

                gen_imgs1 , gen_imgs2 = self.coupled_generators(z)
                validity1, validity2 = self.coupled_discriminators(gen_imgs1, gen_imgs2)

                g_loss = (self.adversarial_loss(validity1,valid)+
                          self.adversarial_loss(validity2, valid))/2

                g_loss.backward()
                self._optimizer_G.step()

                # ----------------------
                #  Train Discriminators
                # ----------------------
                self._optimizer_D.zero_grad()

                validity1_real, validity2_real = self.coupled_discriminators(imgs1,imgs2)
                validity1_fake, validity2_fake = self.coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())

                d_loss = (
                    self.adversarial_loss(validity1_real, valid)
                    +self.adversarial_loss(validity1_fake, fake)
                    +self.adversarial_loss(validity2_real, valid)
                    +self.adversarial_loss(validity2_fake, fake)
                )/4
                d_loss.backward()
                self._optimizer_D.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, i, len(self.mnist), d_loss.item(), g_loss.item())
                )
                if (i + 1) % (len(self.mnist) // 2) == 0:
                    gen_imgs = torch.cat((gen_imgs1.data,gen_imgs2.data),0)
                    show_imgs = (gen_imgs+1)/2
                    show_imgs = show_imgs.to("cpu").numpy()
                    for k, dis_img in enumerate(show_imgs):
                        plt.subplot(8, 8, k + 1)
                        plt.imshow(dis_img.transpose(1,2,0))
                    plt.pause(1)


def main(epochs=200, batch_size=32, latent_dim=100, img_size=32, channels=3, lr=0.0002, b1=0.5,b2=0.999):
    cogan = COGAN_Net(batch_size, latent_dim, img_size, channels, lr, b1,b2)
    cogan.Train(epochs)


if __name__ == "__main__":
    import fire
    fire.Fire(main)