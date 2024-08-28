import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import numpy as np

import math
import os
import warnings

from scripts.scheduler import Scheduler
import scripts.dct_blur as torch_dct


class Degradation:
    
    def __init__(self, timesteps, degradation, noise_schedule, dataset, **kwargs):
        
        self.timesteps = timesteps
        self.device = kwargs['device']
        self.image_size = kwargs['image_size'] 
        self.scheduler = Scheduler(device = self.device)
        
        assert degradation in ['noise', 'blur', 'blur_bansal', 'black', 'black_blur', 'black_blur_bansal', 'black_noise', 'blur_diffusion', 'black_blur_diffusion', 'pixelation', 'black_pixelation'], f'Invalid degradation type, choose from noise, blur, blur_bansal, black, black_blur, black_noise, blur_diffusion. Input: {degradation}'
        self.degradation = degradation
                
        # Denoising 
        self.noise_coefs = DenoisingCoefs(timesteps=timesteps, noise_schedule=noise_schedule, device = self.device)

        # Blacking
        self.blacking_coefs = self.scheduler.get_black_schedule(timesteps = timesteps, mode = 'cosine')


        # Blurring
        blur_kwargs = {'channels': 1 if 'mnist' in dataset else 3, 
                        'kernel_size': 5 if 'mnist' in dataset else 27, # Change to 11 for non-cold start but for conditional sampling (only blurring for 40 steps)
                        'kernel_std': 2 if 'mnist' in dataset else 0.01, # if dataset == 'mnist' else 0.001, # Std has a different interpretation for constant schedule and exponential schedule: constant schedule is the actual std, exponential schedule is the rate of increase # 7 if dataset == 'mnist' else 0.01
                        'timesteps': timesteps, 
                        'blur_routine': 'cifar' if dataset == 'cifar10' else 'constant' if 'mnist' in dataset else 'exponential',
                        'mode': 'circular' if 'mnist' in dataset else 'reflect',
                        'dataset': dataset,
                        'image_size': kwargs['image_size'], 
                        'device': self.device} # if 'mnist' in dataset else 'exponential'} # 'constant' if dataset == 'mnist' else 'exponential'}
            
        self.blur = Blurring(**blur_kwargs)
        
        # Bansal Blurring
        self.blur.get_kernels() # Initialize kernels for Bansal Blurring
        self.blur.gaussian_kernels.to(self.device)  # Move kernels to GPU
        
        # DCT Blurring
        self.dct_blur = self.blur.get_dct_blur() # Initialize DCT Blurring

        # Pixelation 
        if 'pixelation' in degradation:
            assert not timesteps > kwargs['image_size'], 'Number of timesteps must be smaller than the image size for pixelation'
        self.pixel_coefs = self.scheduler.get_pixelation_schedule(timesteps = timesteps, image_size = kwargs['image_size'])
    


    def noising(self, x_0, t, noise = None):
        """
        Function to add noise to an image x at time t, following a common DDPM implementation.
        (https://github.com/lucidrains/denoising-diffusion-pytorch)

        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """

        if noise is None:
            noise = torch.randn_like(x_0, device = self.device)
            warnings.warn('Noise not provided, using random noise')

        x_0_coef, residual_coef = self.noise_coefs.forward_process(t)
        x_0_coef, residual_coef = x_0_coef.to(self.device), residual_coef.to(self.device)
        x_t = x_0_coef * x_0 + residual_coef * noise
        return x_t


    def bansal_blurring(self, x_0, t):
        """
        Function to blur an image x at time t .
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """

        # Freeze kernels
        for kernel in self.blur.gaussian_kernels:
            kernel.requires_grad = False

        x = x_0
        
        # Keep gradients for the original image for backpropagation
        if x_0.requires_grad:
            x.retain_grad()

        t_max = torch.max(t)

        # Blur all images to the max, but store all intermediate blurs for later retrieval         
        max_blurs = []

        if t_max+1 == 0:
            max_blurs.append(x)
        else:
            for i in range(t_max + 1): ## +1 to account for zero indexing of range
                x = x.unsqueeze(0) if len(x.shape) == 2  else x
                x = self.blur.gaussian_kernels[i](x).squeeze(0) 
                if i == (self.timesteps-1):
                    x = torch.mean(x, [2, 3], keepdim=True)
                    x = x.expand(x_0.shape[0], x_0.shape[1], x_0.shape[2], x_0.shape[3])

                max_blurs.append(x)
        
        max_blurs = torch.stack(max_blurs)

        # Choose the correct blur for each image in the batch
        blur_t = []
        for step in range(t.shape[0]):
            if t[step] != -1:
                blur_t.append(max_blurs[t[step], step])
            else:
                blur_t.append(x_0[step])

        return torch.stack(blur_t)
    


    def dct_blurring(self, x_0, t):
        """
        Function to blur an image x at time t using DCT blurring.

        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """

        xt = self.dct_blur(x_0, t).float()
        return xt
    


    def pixelate(self, x, coef):
        """
        Function to pixelate an image x.

        :param torch.Tensor x: The image
        :param int coef: The pixelation coefficient
        :return torch.Tensor: The pixelated image
        """

        og_shape = x.shape
        x = F.interpolate(x, scale_factor=1/coef, mode='nearest')
        x = F.interpolate(x, size=og_shape[2:], mode='nearest')
        return x



    def bansal_pixelation(self, x_0, t):
        """
        Function to pixelate an image x at time t.
        Following the degradation logic of Bansal et al. but using our own pixelation operation.
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """

        t_max = torch.max(t)

        # Pixelate all images to the max, but store all intermediate blurs for later retrieval         
        max_pix = []

        if t_max+1 == 0:
            max_pix.append(x_0)
        else:
            for i in range(t_max + 1): ## +1 to account for zero indexing of range
                x = self.pixelate(x_0, self.pixel_coefs[i])
                max_pix.append(x)
        
        max_pix = torch.stack(max_pix)

        # Choose the correct pixelation for each image in the batch
        pix_t = []
        for step in range(t.shape[0]):
            if t[step] != -1:
                pix_t.append(max_pix[t[step], step])
            else:
                pix_t.append(x_0[step])

        return torch.stack(pix_t)


    def blacking(self, x_0, t):
        """
        Function to fade an image x to black at time t.
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """

        multiplier = self.blacking_coefs[t]
        multiplier[t == -1] = 1.0
        x_t = multiplier * x_0 
        return x_t
    

    
    def degrade(self, x, t, noise = None):
        """
        Function to degrade an image x at time t.
        
        :param x: torch.Tensor
            The image at time t
        :param t: int
            The time step
            
        :return: torch.Tensor
            The degraded image at time t
        """
        if self.degradation == 'noise':
            return self.noising(x, t, noise)
        elif self.degradation == 'black_noise':
            return self.blacking(self.noising(x, t, noise), t)
        elif self.degradation == 'blur':
            return self.dct_blurring(x, t)
        elif self.degradation == 'black':
            return self.blacking(x, t)
        elif self.degradation == 'black_blur':
            return self.blacking(self.dct_blurring(x, t), t)
        elif self.degradation == 'blur_bansal':
            return self.bansal_blurring(x, t)
        elif self.degradation == 'black_blur_bansal':
            return self.blacking(self.bansal_blurring(x, t), t)
        elif self.degradation == 'blur_diffusion':
            return self.blur_diffusion(x, t)
        elif self.degradation == 'black_blur_diffusion':
            return self.blacking(self.blur_diffusion(x, t), t)
        elif self.degradation == 'pixelation':
            return self.bansal_pixelation(x, t)
        elif self.degradation == 'black_pixelation':
            return self.blacking(self.bansal_pixelation(x, t), t)



class Blurring:
    
    def __init__(self, timesteps, channels, image_size, kernel_size, kernel_std, blur_routine, mode, dataset, device):
            """
            Initializes the Blurring class.

            Args:
                channels (int): Number of channels in the input image. Default is 3.
                kernel_size (int): Size of the kernel used for blurring. Default is 11.
                kernel_std (int): Standard deviation of the kernel used for blurring. Default is 7.
                num_timesteps (int): Number of diffusion timesteps. Default is 40.
                blur_routine (str): Routine used for blurring. Default is 'Constant'.
            """


            self.scheduler = Scheduler(device=device)

            self.channels = channels
            self.image_size = image_size
            self.kernel_size = kernel_size
            self.kernel_stds = self.scheduler.get_bansal_blur_schedule(timesteps = timesteps, std = kernel_std, type = blur_routine) 
            self.dct_sigmas = self.scheduler.get_dct_sigmas(timesteps, image_size = image_size)
            self.num_timesteps = timesteps
            self.blur_routine = blur_routine
            self.mode = mode
            self.device = device
        

    def get_conv(self, dims, std, mode):
        """
        Function to obtain a 2D convolutional layer with a Gaussian Blurring kernel.
        
        :param tuple dims: The dimensions of the kernel
        :param tuple std: The standard deviation of the kernel
        :param str mode: The padding mode
        :return nn.Conv2d: The 2D convolutional layer with the Gaussian Blurring kernel
        """
        
        kernel = tgm.image.get_gaussian_kernel2d(dims, std) 
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=dims, padding=int((dims[0]-1)/2), padding_mode=mode,
                         bias=False, groups=self.channels)
         
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        conv.weight = nn.Parameter(kernel, requires_grad=False)

        return conv


    def get_kernels(self):
        """
        Function to obtain a list of 2D convolutional layers with Gaussian Blurring kernels following a certain routine.
        
        :return list: A list of 2D convolutional layers with Gaussian Blurring kernels
        """
        
        kernels = []
        for i in range(self.num_timesteps):
            kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_stds[i], self.kernel_stds[i]), mode = self.mode)) 
        
        self.gaussian_kernels = nn.ModuleList(kernels).to(self.device)
    

    def get_dct_blur(self):
        """
        Function to obtain and initialize the DCT Blur class.

        :return DCTBlur: An instance of the DCT Blur class
        """

        dct_blur = DCTBlur(self.dct_sigmas, self.image_size, self.device)

        return dct_blur
    

class DenoisingCoefs:
    
    def __init__(self, timesteps, noise_schedule, device, **kwargs):
        self.timesteps = timesteps
        self.scheduler = Scheduler(device=device)
        
        self.betas = self.scheduler.get_noise_schedule(self.timesteps, noise_schedule=noise_schedule).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), self.alphas_cumprod[:-1]])
    
    
    def forward_process(self, t):
        """
        Function to obtain the coefficients for the standard Denoising Diffusion process xt = sqrt(alphas_cumprod) * x0 + sqrt(1 - alphas_cumprod) * N(0, I).
        
        :param int t: The time step
        :return tuple: The coefficients for the Denoising Diffusion process
        """

        alpha_t = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        x0_coef = torch.sqrt(alpha_t)
        residual_coef =  torch.sqrt(1. - alpha_t)
        return x0_coef, residual_coef
    
    
    def posterior(self, t):
        """
        Function to obtain the coefficients for the Denoising Diffusion posterior 
        q(x_{t-1} | x_t, x_0).

        :param int t: The time step
        :return tuple: The coefficients for the Denoising Diffusion posterior
        """

        beta_t = self.betas.gather(-1, t).reshape(-1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        alphas_cumprod_prev_t = self.alphas_cumprod_prev.gather(-1, t).reshape(-1, 1, 1, 1)
        
        posterior_variance = beta_t * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t) # beta_hat
        posterior_mean_x0 = beta_t * torch.sqrt(alphas_cumprod_prev_t) / (1. - alphas_cumprod_t) #x_0
        posterior_mean_xt = (1. - alphas_cumprod_prev_t) * torch.sqrt(self.alphas.gather(-1,t).reshape(-1, 1, 1, 1)) / (1. - alphas_cumprod_t) #x_t

        return posterior_mean_x0, posterior_mean_xt, posterior_variance
    
    
    def x0_restore(self, t):
        """
        Function to obtain the coefficients for the Denoising Diffusion reconstruction.
        
        :param int t: The time step
        :return tuple: The coefficients for the Denoising Diffusion process
        """
        
        xt_coef = torch.sqrt(1. / self.alphas_cumprod.gather(-1, t)).reshape(-1, 1, 1, 1)
        residual_coef = torch.sqrt(1. / self.alphas_cumprod.gather(-1, t) - 1).reshape(-1, 1, 1, 1)

        return xt_coef, residual_coef




# DCT Blur Module as in IHDM paper (Risannen et al.) - https://github.com/AaltoML/generative-inverse-heat-dissipation
class DCTBlur(nn.Module):

    def __init__(self, blur_sigmas, image_size, device):
        super(DCTBlur, self).__init__()
        self.blur_sigmas = blur_sigmas.clone().detach().to(device)
        freqs = np.pi*torch.linspace(0, image_size-1,
                                    image_size).to(device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    def forward(self, x, fwd_steps):
        if len(x.shape) == 4:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
        elif len(x.shape) == 3:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None]
        t = sigmas**2/2
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(- self.frequencies_squared * t)
        dct_blurred = torch_dct.idct_2d(dct_coefs, norm='ortho')
        dct_blurred[fwd_steps == -1] = x[fwd_steps == -1] # Keep the original image for t=-1 (needed for Bansal style sampling)
        return dct_blurred




# Custom DCT Blur Module for Bansal Style Sampling
class DCTBlurSampling(nn.Module):

    def __init__(self, blur_sigmas, image_size, device):
        super(DCTBlurSampling, self).__init__()
        self.blur_sigmas = blur_sigmas.clone().detach().to(device)
        freqs = np.pi*torch.linspace(0, image_size-1,
                                    image_size).to(device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    def forward(self, x, fwd_steps = None, t = None):
        if fwd_steps is None:
            sigmas = self.blur_sigmas[:, None, None, None]
        else:
            if len(x.shape) == 4:
                sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
            elif len(x.shape) == 3:
                sigmas = self.blur_sigmas[fwd_steps][:, None, None]
                print(sigmas)
        
        if t is None:
            t = sigmas**2/2
        else:
            pass
            
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(- self.frequencies_squared * t)
        dct_blurred = torch_dct.idct_2d(dct_coefs, norm='ortho')

        if t is None:
            dct_blurred[fwd_steps == -1] = x[fwd_steps == -1] # Keep the original image for t=-1 (needed for Bansal style sampling)
        
            if fwd_steps[0] == -1:
                print("Sampling End reached, returning original image.")

        return dct_blurred
    