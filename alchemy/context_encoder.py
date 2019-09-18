#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: context_encoder.py
@time: 19-9-17 下午10:22
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
from dataset.mini_imagenet import CelebA
from net.context_encoder import *
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 12.0)

class ContextEncoder(object):
    def __init__(self, batch_size,img_size, mask_size, lr, b1, b2):
        self.img_size = img_size
        self.mask_size = mask_size
        #Calculate output of image discriminator(PatchGAN)
        patch_h, patch_w = int(self.mask_size/2**3), int(self.mask_size/2**3)
        self.patch = (1, patch_h, patch_w)
        self.adversarial_loss = torch.nn.MSELoss()
        self.pixelwise_loss = torch.nn.L1Loss()

        self.generator = Generator(channels=3)
        self.discriminator = Discriminator(channels=3)
        self.generator.apply(self.weights_init_normal)
        self.discriminator.apply(self.weights_init_normal)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        celeba = CelebA()
        self.train_loader = celeba.get_loader(batch_size,
                                   mode="train",
                                   img_size=img_size,
                                   mask_size=mask_size)
        self.test_loader = celeba.get_loader(12,
                                              mode="val",
                                              num_workers=1,
                                              img_size=img_size,
                                              mask_size=mask_size)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr,betas=(b1,b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1,b2))


    @staticmethod
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def sample_images(self):
        samples, masked_samples, i = next(iter(self.test_loader))
        samples = samples.to(self.device)
        masked_samples = masked_samples.to(self.device)
        i = i[0].item() #Upper-left coordinate of mask
        #Generate inpainted image
        gen_mask = self.generator(masked_samples)
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + self.mask_size, i : i + self.mask_size] = gen_mask
        sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
        #save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)

        dis_sample = sample.data.to("cpu")
        dis_sample = dis_sample.numpy()

        for k, dis_img in enumerate(dis_sample):
            dis_img = (dis_img+1)/2
            plt.subplot(2,6,k+1)
            plt.imshow(dis_img.transpose(1,2,0))
        plt.pause(1)


    def Train(self, epochs):
        for epoch in range(epochs):
            for i, (imgs, masked_imgs, masked_parts) in enumerate(self.train_loader):
                #Adversarial ground truths
                valid = torch.ones(imgs.shape[0],*self.patch).to(self.device)
                fake = torch.zeros(imgs.shape[0],*self.patch).to(self.device)

                # Configure input
                imgs = imgs.to(self.device)
                masked_imgs = masked_imgs.to(self.device)
                masked_parts = masked_parts.to(self.device)

                #-------------
                # Train Generator
                #--------------

                self.optimizer_G.zero_grad()
                gen_parts = self.generator(masked_imgs)
                g_adv = self.adversarial_loss(self.discriminator(gen_parts),valid)
                g_pixel = self.pixelwise_loss(gen_parts, masked_parts)
                g_loss = 0.001 * g_adv + 0.999 * g_pixel

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(masked_parts), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_parts.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                    % (epoch, epochs, i, len(self.train_loader), d_loss.item(), g_adv.item(), g_pixel.item())
                )
                if (i + 1) % (len(self.train_loader) // 2) == 0:
                    self.sample_images()


def main(epochs=20, batch_size=16,img_size=128, mask_size=64, lr=0.0002, b1=0.5, b2=0.999):
    context_encoder = ContextEncoder(batch_size, img_size, mask_size, lr, b1, b2)
    context_encoder.Train(epochs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)