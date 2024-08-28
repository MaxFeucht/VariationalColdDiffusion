import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import numpy as np

import math
import os

import scripts.dct_blur as torch_dct


class Scheduler:
    
    def __init__(self, **kwargs):
        self.device = kwargs['device']


    def linear(self, timesteps):  # Problematic when using < 20 timesteps, as betas are then surpassing 1.0
        """
        Linear schedule, proposed in original ddpm paper

        :param int timesteps: The number of timesteps
        :return torch.Tensor: The linear schedule
        """
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)


    def cosine(self, timesteps, s = 0.008, black = False):
        """
        Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ

        :param int timesteps: The number of timesteps
        :param float s: The cosine schedule parameter
        :param bool black: Whether to return the blacking schedule
        :return torch.Tensor: The cosine schedule
        """

        steps = timesteps + 1 if not black else timesteps
        t = torch.linspace(0, timesteps, steps, dtype = torch.float32) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        
        # If we want the blacking schedule, we return the alphas_cumprod
        if black == True:
            return alphas_cumprod
        
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
        

    def sigmoid(self, timesteps, start = -3, end = 3, tau = 1):
        """
        Sigmoid schedule as proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        Better for images > 64x64, when used during training

        :param int timesteps: The number of timesteps
        :param float start: The start value of the sigmoid
        :param float end: The end value of the sigmoid
        :param float tau: The tau value of the sigmoid
        :return torch.Tensor: The sigmoid schedule
        """

        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float32) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)


    def get_noise_schedule(self, timesteps, noise_schedule):
        """
        Function to obtain the noise schedule for the Denoising Diffusion process.

        :param int timesteps: The number of timesteps
        :param str noise_schedule: The noise schedule type
        :return torch.Tensor: The noise schedule
        """

        if noise_schedule == 'linear':
            return self.linear(timesteps)
        elif noise_schedule == 'cosine':
            return self.cosine(timesteps)
        elif noise_schedule == 'sigmoid':
            return self.sigmoid(timesteps)
        else:
            raise ValueError('Invalid schedule type')

    
    def get_bansal_blur_schedule(self, timesteps, std = 0.01, type = 'exponential'):
        """
        Function to obtain the standard deviation schedule for Blurring as performed in Bansal et al. (https://github.com/arpitbansal297/Cold-Diffusion-Models)

        :param int timesteps: The number of timesteps
        :param float std: The standard deviation of the kernel
        :param str type: The schedule type
        :return torch.Tensor: The standard deviation schedule
        """

        # Blur schedules as implemented in the original code by Bansal et al.      
        if type == 'constant':
            return torch.ones(timesteps, dtype=torch.float32) * std

        if type == 'exponential':
            return torch.exp(std * torch.arange(timesteps, dtype=torch.float32))
        
        if type == 'cifar':
            return torch.arange(timesteps, dtype=torch.float32)/100 + 0.35
    

    def get_dct_sigmas(self, timesteps, image_size): 
        """
        Function to obtain the sigma schedule for DCT Blurring.

        :param int timesteps: The number of timesteps
        :param int image_size: The size of the image
        :return torch.Tensor: The sigma schedule
        """

        dct_sigma_min = 1
        dct_sigma_max = image_size
        
        dct_sigmas = torch.exp(torch.linspace(np.log(dct_sigma_min),
                                                np.log(dct_sigma_max), timesteps-1, device=self.device))
        
        # Repeat last sigma, to have max sigma for the first non-blacked image
        dct_sigmas = torch.cat((dct_sigmas, torch.ones(1).to(self.device) * dct_sigmas[-1]))

        if timesteps == 1:
            dct_sigmas = torch.tensor([dct_sigma_max], device=self.device, dtype = torch.float32)

        return dct_sigmas



    def get_black_schedule(self, timesteps, mode, factor = 0.95):
        """
        Function to obtain the blacking schedule for a forward that involves blacking.

        :param int timesteps: The number of timesteps
        :param str mode: The blacking schedule type
        :param float factor: The factor for the blacking schedule (only for exponential)
        :return torch.Tensor: The blacking schedule coefficients
        """
        
        t_range = torch.arange(timesteps, dtype=torch.float32)

        if mode == 'linear':
            coefs = (1 - (t_range+1) / (timesteps))  # +1 bc of zero indexing
        
        elif mode == 'exponential':
            coefs = factor ** (t_range)  
        
        elif mode == 'cosine':
            coefs = self.cosine(timesteps, black=True)
        
        # Explicitly set the last value to 0 for complete blacking
        coefs[-1] = 0.0
            
        return coefs.reshape(-1, 1, 1, 1).to(self.device)


    
    def get_pixelation_schedule(self, timesteps, image_size):
        """
        Function to obtain the schedule for a pixelation forward.

        :param int timesteps: The number of timesteps
        :param int image_size: The size of the image
        :return list: The pixelation schedule
        """

        pixel_coefs = torch.exp(torch.linspace(np.log(1.5), np.log(image_size/2 + 5), timesteps)) # log scale coefs to have smooth transitions in beginning
        pixel_coefs = [int(pix) for pix in pixel_coefs] # int for compatibility / efficiency with F.interpolate
        return pixel_coefs
