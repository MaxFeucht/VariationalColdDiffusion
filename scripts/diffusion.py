import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import numpy as np
from sklearn.mixture import GaussianMixture

import math
import os
import copy
import warnings
from tqdm import tqdm

import scripts.dct_blur as torch_dct
from scripts.ema import ExponentialMovingAverage



class Scheduler:
    def __init__(self, **kwargs):
        self.device = kwargs['device']

    def linear(self, timesteps):  # Problematic when using < 20 timesteps, as betas are then surpassing 1.0
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

    def cosine(self, timesteps, s = 0.008, black = False):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
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
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
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
        if noise_schedule == 'linear':
            return self.linear(timesteps)
        elif noise_schedule == 'cosine':
            return self.cosine(timesteps)
        elif noise_schedule == 'sigmoid':
            return self.sigmoid(timesteps)
        else:
            raise ValueError('Invalid schedule type')

    
    def get_bansal_blur_schedule(self, timesteps, std = 0.01, type = 'exponential'):


        # The standard deviation of the kernel starts at 1 and increases exponentially at the rate of 0.01.        
        if type == 'constant':
            return torch.ones(timesteps, dtype=torch.float32) * std

        if type == 'exponential':
            return torch.exp(std * torch.arange(timesteps, dtype=torch.float32))
        
        if type == 'cifar':
            return torch.arange(timesteps, dtype=torch.float32)/100 + 0.35
    

    def get_dct_sigmas(self, timesteps, image_size): 

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
        
        t_range = torch.arange(timesteps, dtype=torch.float32)

        if mode == 'linear':
            coefs = (1 - (t_range+1) / (timesteps))  # +1 bc of zero indexing
        
        elif mode == 'exponential':
            coefs = factor ** (t_range)  
        
        elif mode == 'cosine':
            coefs = self.cosine(timesteps, black=True)
        
        # Explicitly set the last value to 0
        coefs[-1] = 0.0
            
        return coefs.reshape(-1, 1, 1, 1).to(self.device)

        
        
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
        self.pixel_coefs = torch.exp(torch.linspace(np.log(1.5), np.log(kwargs['image_size']/2 + 5), 10)) # log scale coefs to have smooth transitions in beginning
        self.pixel_coefs = [int(pix) for pix in self.pixel_coefs] # int for compatibility / efficiency with F.interpolate
    

    def noising(self, x_0, t, noise = None):
        """
        Function to add noise to an image x at time t.
        
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
        Function to blur an image x at time t.
        
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
    

    def bansal_blackblurring_xt(self, x_tm1, t):
        """
        Function to blur an image x at time t.
        
        :param torch.Tensor x_tm1: Degraded image at time t-1
        :param int t: The time step
        :return torch.Tensor x_t: The degraded image at time t
        """

        x = x_tm1

        t_max = torch.max(t)

        if t_max == -1:
            return x_tm1
        else:
            # Blur all t that are not max t (Error otherwise)
            x = x.unsqueeze(0) if len(x.shape) == 2  else x
            x_t = self.blur.gaussian_kernels[t_max](x).squeeze(0)  
            
            # Blacking just for one step
            mult_tm1 = self.blacking_coefs[t-1] if t_max-1 != -1 else 1.0
            mult_t = self.blacking_coefs[t]
            mult = mult_t / mult_tm1
            x_t = mult * x_t 

            return x_t
                

    def dct_blurring(self, x_0, t):
        xt = self.dct_blur(x_0, t).float()
        return xt
    

    def blur_diffusion(self, x_0, t, s = 0.001, noise = None):
    
        if noise is None:
                noise = torch.randn_like(x_0, device = self.device)
                warnings.warn('Noise not provided, using random noise')
        
        # Noise Coefficients
        a_t = (torch.cos((t + s) / (1 + s) * math.pi * 0.5)**2).to(self.device)
        sig_t = 1 - a_t

        # Blur Coefficients
        #max_blur_sigma = self.dct_blur.blur_sigmas[-1] * 10 # In Hoogeboom a tunable hyperparameter
        max_blur_sigma = self.image_size/2
        sig_B_t = (max_blur_sigma * sig_t)[:, None, None, None]
        tau = sig_B_t**2/2
        dt = (1 - 0.001) * torch.exp(-self.dct_blur.frequencies_squared * tau) + 0.001

        # Combined Coefficients
        alpha = torch.sqrt(a_t[:, None, None, None]) * dt
        sigma = sig_B_t # Potentially has to be expanded to the correct shape

        # V_Transpose
        freq_data = torch_dct.dct_2d(x_0, norm='ortho')
        freq_noise = torch_dct.dct_2d(noise, norm='ortho')
        freq_latent = alpha * freq_data + sigma * freq_noise

        # V
        latent = torch_dct.idct_2d(freq_latent, norm='ortho')

        return latent #, freq_noise


    def pixelate(self, x, coef):
        og_shape = x.shape
        x = F.interpolate(x, scale_factor=1/coef, mode='nearest')
        x = F.interpolate(x, size=og_shape[2:], mode='nearest')
        return x



    def bansal_pixelation(self, x_0, t):
        """
        Function to pixelate an image x at time t.
        
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
        
        :param x: torch.Tensor
            The image at time t
        :param t: int
            The time step
            
        :return: tuple
            The coefficients for the Denoising Diffusion process
        """
        alpha_t = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        x0_coef = torch.sqrt(alpha_t)
        residual_coef =  torch.sqrt(1. - alpha_t)
        return x0_coef, residual_coef
    
    
    def posterior(self, t):
        """
        Function to obtain the coefficients for the Denoising Diffusion posterior 
        q(x_{t-1} | x_t, x_0).
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

class Reconstruction:
    
    def __init__(self, prediction, degradation, **kwargs):
        self.prediction = prediction
        self.deterministic = False if degradation in ['noise'] else True
        self.coefs = DenoisingCoefs(**kwargs)

    def reform_pred(self, model_output, x_t, t, return_x0 = False):
        """
        Function to reform predictions for a given degraded image x_t at time t, using the output of a trained model and a degradation function.
        
        :param torch.Tensor model_output: The output of the model - either the residual or the x0 estimate
        :param torch.Tensor x_t: The degraded image at time t
        :param int t: The time step
        :return torch.Tensor: The predicted image at time t-1
        """
        
        if not self.deterministic: 
            xt_coef, residual_coef = self.coefs.x0_restore(t) # Get coefficients for the Denoising Diffusion process
        else:
            xt_coef, residual_coef = torch.tensor(1.), torch.tensor(1.) # Coefficients are 1 for deterministic degradation
                
        if self.prediction == 'residual':
            residual = model_output
            if not return_x0:
                return residual
            else:
                x0_estimate = xt_coef * x_t - residual_coef * residual 
                return x0_estimate     

        elif self.prediction == 'x0':
            x0_estimate = model_output
            if return_x0:
                return x0_estimate
            else:
                residual = (xt_coef * x_t - x0_estimate) / residual_coef
                return residual      
        elif 'xt' in self.prediction:
            return model_output
        else:
            raise ValueError('Invalid prediction type')
        

class Loss:
    
    def __init__(self, **kwargs):
        self.degradation = Degradation(**kwargs)
    
    def mse_loss(self, target, pred):    
        return F.mse_loss(pred, target, reduction='mean')
    
    def cold_loss(self, target, pred, t):
        diff = pred - target
        return diff.abs().mean()  # Mean over batch dimension

    def darras_loss(self, target, pred, t):
        diff = pred - target # difference between the predicted and the target image / residual
        degraded_diff = self.degradation.degrade(diff, t)**2 # Move this difference into the degradation space and square it
        return degraded_diff.mean()  # Squared error loss, averaged over batch dimension



class Trainer:
    
    def __init__(self, model, lr, timesteps, prediction, degradation, noise_schedule, vae, vae_beta, **kwargs):

        self.device = kwargs['device']
        self.model = model.to(self.device)
        self.prediction = prediction
        self.timesteps = timesteps
        self.deterministic = False if degradation in ['noise'] else True
        self.black = True if 'black' in degradation else False
        self.vae = vae
        self.vae_beta = vae_beta
        self.noise_scale = kwargs['noise_scale']
        self.ris_noise = kwargs['ris_noise']
        self.loss_weighting = kwargs['loss_weighting']
        self.min_t2_step = kwargs['min_t2_step']
        self.kwargs = kwargs
        self.add_noise = kwargs['vwd']
        self.autoencoder = kwargs['autoencoder']
        self.vcd = kwargs['vcd']
        self.perturb_counter = 0

        if self.vcd:
            self.teacher = copy.deepcopy(model)
            self.teacher.eval()


        general_kwargs = {'timesteps': timesteps, 
                          'prediction': prediction,
                          'degradation': degradation, 
                          'noise_schedule': noise_schedule, 
                          'device': self.device,
                          'dataset': kwargs['dataset'],
                          'image_size': kwargs['image_size']}
        
        self.schedule = Scheduler(device=self.device)
        self.degradation = Degradation(**general_kwargs)
        self.reconstruction = Reconstruction(**general_kwargs)
        self.loss = Loss(**general_kwargs)
        self.annealer = self.schedule.cosine(kwargs['kl_annealing'], black = self.black)
        self.anneal_iters = kwargs['kl_annealing']
        self.anneal_counter = 0
        self.cold_warmup_passed = False
        self.cyclic_annealing = False#True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))
        #self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=0.3, patience=30)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.75) # Halve learning rate ever 300 epochs
        self.apply_ema = not kwargs['skip_ema']
        self.test_run = True if kwargs['test_run'] else False

        # Define Model EMA
        if self.apply_ema:
            self.model_ema = ExponentialMovingAverage(self.model.parameters(), decay=kwargs['model_ema_decay'])
        else:
            self.model_ema = model
            warnings.warn('No EMA applied')


    def train_iter(self, x_0, t, t2=None, noise=None):

        # Degrade and obtain residual
        if not self.deterministic:
            noise = torch.randn_like(x_0, device=self.device) # Important: Noise to degrade with must be the same as the noise that should be predicted

        x_t = self.degradation.degrade(x_0, t, noise=noise) 
        if self.prediction == 'vxt':
            assert t2 is not None, "Second timesteps must be supplied for Variable Timestep Diffusion"
            vxt = self.degradation.degrade(x_0, t2, noise=noise) #x_t- = x_t2 with t2 < t
        elif self.prediction == 'xt':
            x_tm = self.degradation.degrade(x_0, t-self.min_t2_step, noise=noise) # x_t- = x_t-1

        # Add noise to degrade with - Noise injection a la Risannen
        if self.ris_noise:
            x_t = x_t + torch.randn_like(x_0, device=self.device) * self.noise_scale

        if not self.deterministic:
            residual = noise 


        # Define prediction and target
        if self.prediction == 'residual':
            assert not self.deterministic, 'Residual prediction only possible for Denoising Diffusion'
            target = residual
            ret_x0 = False
        elif self.prediction == 'x0':
            target = x_0
            ret_x0 = True
        elif self.prediction == 'xt':
            target = x_tm
            ret_x0 = False
        elif self.prediction == 'vxt':
            target = vxt
            ret_x0 = False
        else:
            raise ValueError('Invalid prediction type')
        

        # Get Model prediction with correct output and select appropriate loss
        if self.vae: 
            
            consistency = False

            # Cold bootstrapping perturbation using a teacher model that inherits the weights of the EMA model, parallel to sampling
            # Using EMA to stabilize the teacher model, as single model predictions are highly fluctuating
            if self.vcd:                  
                
                # Fix random seed for perturbation dropout
                torch.manual_seed(torch.randint(0, 100000, (1,)).item())

                p = 0.125
                if self.anneal_counter > 500:
                    self.model_ema.copy_to(self.teacher.parameters())
                else:
                    self.teacher.load_state_dict(self.model.state_dict())
                    p = 0.3
                
                if self.perturb_counter == 0:

                    # Generate the full (conditional) sampling trajectory, then select the correct prediction for each image in the batch
                    with torch.no_grad():

                        T = torch.ones_like(t) * int(self.timesteps-1)
                        x_T = self.degradation.degrade(x_0, T, noise=noise) # x_t2+1

                        model_preds = []
                        x_t_hat = x_T

                        for i in range(self.timesteps-1, -1, -1): ## +1 to account for zero indexing of range  
                            
                            tp = torch.ones_like(t) * i
                            t2p = tp - self.min_t2_step
                            cond_tp = self.degradation.degrade(x_0, t2p, noise=noise) # x_t2+1

                            if torch.rand(1).item() < p: 
                                x_t_hat = self.degradation.degrade(x_0, tp, noise=noise) # Lead back to the good path, equally probable at all steps

                            pred_hat, _ = self.teacher(x_t_hat, tp, cond_tp, t2=t2p)
         
                            x_t_hat = x_t_hat + pred_hat
                            model_preds.append(x_t_hat)

                        model_preds = torch.stack(model_preds)

                        # Choose the correct model pred for each image in the batch
                        x_t_prime = []
                        for step in range(t.shape[0]):
                            idx = self.timesteps-1 - t[step] # If t = 9, we want the first element of the model_preds, i.e., idx = 0
                            x_t_prime.append(model_preds[idx, step])

                        x_t_prime = torch.stack(x_t_prime)

                        self.perturb_counter = 0
                else:
                    self.perturb_counter += 1

            if not consistency and self.vcd:
                x_t = x_t_prime

            # Condition VAE on target
            cond = target
            model_pred, kl_div = self.model(x_t, t, cond, t2=t2) # VAE Model needs conditioning signal for prediction

            # Testing to include VAE Noise into Loss, just as in Risannen. 
            # We do this by adding the noise to x_t and let the model optimize for the difference between the perturbed x_t and xtm1.
            if self.kwargs['vae_loc'] == 'bold':
                x_t = x_t + self.model.vae_noise 

            # Risannen Loss Formulation - Here the slightly perturbed x_t is used for the prediction
            if self.prediction in ['xt', 'vxt']: # Check if that really helps for vxt
                pred = (x_t + model_pred)
                #pred = model_pred
            else:
                pred = model_pred

            # Sum over all pixels before averaging, important for loss scaling, analog to the original Risannen implementation
            if self.prediction == 'xt' or self.prediction == 'vxt':
                reconstruction = (target - pred)**2 # L2 penalty 
                reconstruction = torch.sum(reconstruction.reshape(reconstruction.shape[0], -1), dim=-1).mean() 

                # CP Addition
                if consistency and self.vcd:
                    cp = (x_t_prime - pred)**2
                    cp = torch.sum(cp.reshape(cp.shape[0], -1), dim=-1).mean()
                    reconstruction = reconstruction + cp * 0.05 # Add the consistency property to the reconstruction loss

            else:
                reconstruction = self.loss.mse_loss(target, pred)
                
            if self.loss_weighting:
                if t2 is not None:
                    weight = self.degradation.blacking_coefs[t2].reshape(-1, 1, 1, 1)
                else:
                    weight = self.degradation.blacking_coefs[t-1].reshape(-1, 1, 1, 1)
                weight[t2 == -1] = 1.0
                weight = weight.mean().item()
                reconstruction = reconstruction / weight # Weighting of the reconstruction loss image according to the time step - the higher the time step, the more the loss is upweighted

            if not self.autoencoder:
                
                if self.anneal_iters == 0:
                    self.annealing_factor = 1.0
                else:
                    # Set KL to almost 0 for the first 50% of the annealing steps, only then start annealing
                    if not self.cold_warmup_passed and self.anneal_counter == int(self.anneal_iters / 3):
                        self.cold_warmup_passed = True
                        self.anneal_counter = 0

                    if not self.cold_warmup_passed: 
                        self.annealing_factor = 1 - self.annealer[0]

                    else:
                        # KL Annealing (applied in train_iter)
                        if self.anneal_counter < self.anneal_iters:
                            self.annealing_factor = 1 - self.annealer[self.anneal_counter]
                        else:
                            self.annealing_factor = 1.0
                                        
                        # Reset the annealing counter to start new cycle
                        if self.cyclic_annealing:
                            if self.anneal_counter == self.anneal_iters * 2:
                                self.anneal_counter = 0
                                self.cold_warmup_passed = False # Reset cold warmup for each cycle
                                print('Annealing cycle reset')
                        
                    self.anneal_counter += 1

                loss = reconstruction + self.vae_beta * kl_div * self.annealing_factor #* self.noise_scale)
                return loss, reconstruction, kl_div
            else:
                return reconstruction
        
        else:

            model_pred = self.model(x_t, t, t2=t2) # Model prediction without VAE 

            # Risannen Loss Formulation - Here the slightly perturbed x_t is used for the prediction
            if self.prediction in ['xt', 'vxt']: # Check if that really helps for vxt
                pred = (x_t + model_pred)
            else:
                pred = model_pred

            if not self.deterministic:
                pred = self.reconstruction.reform_pred(pred, x_t, t, return_x0=ret_x0) # Model prediction in correct form with coefficients applied
            
            if self.prediction == "x0" and not self.ris_noise:
                loss = (target - pred).abs() # L1 penalty for Bansal
            else:
                loss = (target - pred)**2 # L2 penalty for everything else

            # Sum over all pixels before averaging, important for loss scaling
            loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1).mean() 

            #loss = loss.mean()
            return loss
    
    
    def train_epoch(self, dataloader, val = False):
        
        # Set model to train mode
        if not val:
            assert self.model.train(), 'Model not in training mode'
        else:
            assert self.model.eval(), 'Model not in evaluation mode'

        # Iterate through trainloader
        epoch_loss = 0  
        epoch_reconstruction = 0
        epoch_kl_div = 0
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x_0, _ = data
            x_0 = x_0.to(self.device)
            
            # Sample t
            t = torch.randint(0, self.timesteps, (x_0.shape[0],), dtype=torch.long, device=self.device) # Randomly sample time steps

            # Sample t2 for variable xt prediction 
            if self.prediction == 'vxt':
                t2 = []
                for t_ in t: # Select a second timestep on a sparse grid between -1 and t, in steps of min_t2_step from t
                    x = (t_ // self.min_t2_step) + 1 # + 1 to be able to go to x0 from t
                    diff = torch.randint(1, x + 1, (1,)).item() # +1 for indexing of torch.randint
                    t2_ = t_ -  diff * self.min_t2_step
                    t2.append(t2_)
            
                t2 = torch.clamp(torch.tensor(t2, dtype=torch.long, device=self.device), min=-1)


                # NEW VAR T INTERPRETATION
                t2 = t - self.min_t2_step
                t2 = torch.clamp(t2, min=-1)

            elif self.prediction == 'xt':
                t2 = t - self.min_t2_step
            else:
                t2 = None

            if self.vae and not self.autoencoder:
                loss, reconstruction, kl_div = self.train_iter(x_0, t, t2=t2)
                epoch_reconstruction += reconstruction.item()
                epoch_kl_div += kl_div.sum().item()

            else:
                loss = self.train_iter(x_0, t, t2=t2)
            
            epoch_loss += loss.sum().item()

            # To Do: Implement Gradient Accumulation
            if not val:

                loss.sum().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update EMA
                if self.apply_ema:
                    self.model_ema.update(self.model.parameters())
            
            # Break prematurely if args.test_run
            if self.test_run:
                break
        
        # Update LR Scheduler 
        if not self.vae or self.autoencoder: # for non-VAE setting
            self.lr_scheduler.step()
            print('Last LR: ', self.lr_scheduler.get_last_lr())
        elif self.vae and not self.autoencoder: # for VAE setting
            self.lr_scheduler.step()
            print('Last LR: ', self.lr_scheduler.get_last_lr())

        if self.vae and not self.autoencoder:
            return epoch_loss/len(dataloader), epoch_reconstruction/len(dataloader), epoch_kl_div/len(dataloader)
        else:
            return epoch_loss/len(dataloader)

     
class Sampler:
    
    def __init__(self, timesteps, prediction, noise_schedule, degradation, **kwargs):
        self.degradation = Degradation(timesteps=timesteps, degradation=degradation, prediction=prediction, noise_schedule=noise_schedule, **kwargs)
        self.reconstruction = Reconstruction(timesteps=timesteps, prediction=prediction, degradation = degradation, noise_schedule=noise_schedule, **kwargs)
        self.prediction = prediction
        self.timesteps = timesteps
        self.device = kwargs['device']
        self.deterministic = False if degradation in ['noise'] else True
        self.black = True if 'black' in degradation else False
        self.gmm = None
        self.ris_noise = kwargs['ris_noise']
        self.break_symmetry = kwargs['break_symmetry']
        self.vae = kwargs['vae']
        self.noise_scale = kwargs['noise_scale']
        self.loss_weighting = kwargs['loss_weighting']
        self.kwargs = kwargs
        self.add_noise = kwargs['vwd']
        self.tacos = False if self.add_noise else True



    def fit_gmm(self, dataloader, clusters = 1, sample = False):
        """
        Function to fit a Gaussian Mixture Model to the mean of data in the dataloader. 
        Used to sample cold start images for deblurring diffusion.

        :param GMM: The Gaussian Mixture Model class
        :param DataLoader dataloader: The dataloader containing the data
        :param int clusters: The number of clusters in the Gaussian Mixture Model
        """

        # Fit GMM for cold sampling
        all_samples = None # Initialize all_samples
        for i, data in enumerate(dataloader, 0):
            img, _ = data
            img = torch.mean(img, [2, 3])
            if all_samples is None: 
                all_samples = img
            else:
                all_samples = torch.cat((all_samples, img), dim=0)

        all_samples = all_samples

        self.gmm = GaussianMixture(n_components=clusters, covariance_type='full', tol = 0.001)

        self.gmm.fit(all_samples)
        print("GMM fitted")


    def sample_x_T(self, batch_size, channels, image_size):
        """
        Function to sample x_T either from a Gaussian Mixture Model or from a random normal distribution.

        :param int batch_size: The batch size of the samples
        :param int channels: The number of channels in the samples
        :param int image_size: The size of the images in the samples
        """

        if self.deterministic and not self.black:

            # Sample x_T from GMM
            if self.gmm is None:
                raise ValueError('GMM not fitted, please fit GMM before cold sampling')
            else:
                assert isinstance(self.gmm, GaussianMixture), 'GMM not fitted correctly'
                channel_means = self.gmm.sample(n_samples=batch_size)[0] # Sample from GMM
                channel_means = torch.tensor(channel_means, dtype=torch.float32, device=self.device)
                channel_means = channel_means.unsqueeze(2).unsqueeze(3)
                x_t = channel_means.expand(batch_size, channels, image_size, image_size) # Expand the channel-wise means to the correct dimensions to build x_T
                x_t = x_t.float()
                    
            # Noise injection for breaking symmetry
            if self.ris_noise:
                x_t = x_t + torch.randn_like(x_t, device=self.device) * self.noise_scale
        
        elif self.black:
            # Sample x_T from R0
            x_t = torch.zeros((batch_size, channels, image_size, image_size), device=self.device) 

            # For breaking Symmetry in Bansal Baseline
            if self.break_symmetry:
                x_t = x_t + torch.randn_like(x_t, device=self.device) * self.noise_scale

        elif not self.deterministic and not self.black:
            # Sample x_T from random normal distribution
            x_t = torch.randn((batch_size, channels, image_size, image_size), device=self.device)
        
        self.x_T = x_t

        print("x_T sampled and fixed")


    @torch.no_grad() 
    def sample_ddpm(self, model, batch_size):

        model.eval()

        # Sample x_T either every time new or once and keep it fixed 
        if self.x_T is None:
            x_t = self.sample_x_T(batch_size, model.channels, model.image_size)
        else:
            x_t = self.x_T
        
        ret_x_T = x_t

        samples = []
        for t in tqdm(reversed(range(self.timesteps)), desc="DDPM Sampling"):
            t_tensor = torch.full((batch_size,), t).to(self.device)
            z = torch.randn((batch_size, model.in_channels, model.image_size, model.image_size)).to(self.device)
            posterior_mean_x0, posterior_mean_xt, posterior_var = self.reconstruction.coefs.posterior(t_tensor) # Get coefficients for the posterior distribution q(x_{t-1} | x_t, x_0)
            model_pred = model(x_t, t_tensor)
            x_0_hat = self.reconstruction.reform_pred(model_pred, x_t, t_tensor, return_x0 = True) # Obtain the estimate of x_0 at time t to sample from the posterior distribution q(x_{t-1} | x_t, x_0)
            x_0_hat.clamp_(-1, 1) # Clip the estimate to the range [-1, 1]
            x_t_m1 = posterior_mean_xt * x_t + posterior_mean_x0 * x_0_hat + torch.sqrt(posterior_var) * z # Sample x_{t-1} from the posterior distribution q(x_{t-1} | x_t, x_0)

            x_t = x_t_m1
            samples.append(x_t) 

        return samples, ret_x_T
 

    @torch.no_grad()
    def sample_cold(self, model, batch_size = 16, x0=None, generate=False, prior=None, t_inject=None, t2=None, t_diff=1):

        model.eval()

        t=self.timesteps

        # Sample x_T either every time new or once and keep it fixed 
        if generate:
            if self.x_T is None:
                xT = self.sample_x_T(batch_size, model.channels, model.image_size)
            else:
                xT = self.x_T
            cond = None
        else:
            t_tensor = torch.full((batch_size,), t-1, dtype=torch.long).to(self.device) # t-1 to account for 0 indexing and the resulting t+1 in the degradation operation
            xT = self.degradation.degrade(x0, t_tensor) # Adaption due to explanation below (0 indexing)
            cond = x0

        xt = xT

        direct_recons = None
        sampling_noise = None
        samples = []
        samples.append(xT) 
        steps = t_diff if t_diff > 0 else 1

        if t_diff != 1:
            print(f"Sampling with t_diff = {t_diff} and steps = {steps}")
        
        # Changing Sampling Noise every time we sample, but not in loop 
        if self.add_noise and self.prediction != 'x0':
            model.sample_noise = torch.randn_like(xt).to(xt.device) * self.noise_scale


        for t in tqdm(reversed(range(0, self.timesteps, steps)), desc=f"Cold Sampling"):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long).to(self.device) # t-1 to account for 0 indexing that the model is seeing during training
            
            # Sample t2 for variable xt prediction
            if self.prediction in ['xt', 'vxt']:
                if t_diff <= 0: # Equals to x0 prediction
                    t2 = torch.full((batch_size,), -1, dtype=torch.long).to(self.device) # t-1 to account for 0 indexing that the model is seeing during training
                elif t_diff > 0: # Equals to xt prediction of variable timestep
                    t2 = torch.full((batch_size,), t-t_diff, dtype=torch.long).to(self.device)
                if not generate:
                    cond = self.degradation.degrade(x0, t2) # Condition on the true less degraded image                    

            if self.vae:
                pred, _ = model(xt, t_tensor, cond=cond, prior=prior, t2=t2)
            else:
                pred = model(xt, t_tensor, t2=t2)


            # BANSAL ALGORITHM 2
            if self.prediction == 'x0' or (self.prediction == 'vxt' and t_diff == -1):

                # Remove sampling noise from x0 sampling, AFTER it was used for prediction
                if sampling_noise is not None:
                    xt = xt - sampling_noise 
                
                x0_hat = (xt + pred) if self.prediction == 'vxt' else pred 

                if self.tacos:
                    xt_hat = self.degradation.degrade(x0_hat, t_tensor)
                    xtm1_hat = self.degradation.degrade(x0_hat, t_tensor - 1) # This returns x0_hat for t=0
                    xtm1 = xt - xt_hat + xtm1_hat
                else:
                    xt_hat = self.degradation.degrade(x0_hat, t_tensor)
                    xtm1_hat = self.degradation.degrade(x0_hat, t_tensor - 1) # This returns x0_hat for t=0
                    xtm1 = xt - xt_hat + xtm1_hat

                if direct_recons == None:
                    direct_recons = x0_hat
    

            # OURS with xt prediction (includes variable timestep prediction)
            elif self.prediction == 'xt' or (self.prediction == 'vxt' and t_diff != -1):

                # Risannen Analogue
                if self.kwargs['vae_loc'] == 'bold':
                    xt = xt + model.vae_noise #* 1.25

                xtm1 = (xt + pred) if (self.ris_noise or self.add_noise or self.vae) else pred # According to Risannen, residual prediction stabilizes the training

            # OURS with residual prediction
            elif self.prediction == 'residual':
                residual = pred
                xtm1 = xt + residual

            # In Risannen the noise is added to the predicted image, AFTER the model prediction
            if self.ris_noise and not t == 0:
                sampling_noise = torch.randn_like(xt, device=self.device) * self.noise_scale * 1.25 # 1.25 is a scaling factor from the original Risannen Code (delta = 1.25 * sigma)
                xtm1 = xtm1 + sampling_noise   
                     
            # Change from xtm1 to xt for next iteration
            xt = xtm1
            samples.append(xt)

        return samples, xT, direct_recons, xt


    def sample(self, model, batch_size, x0=None, prior=None, generate=False, t_inject=None, t_diff=1):
        
        if self.deterministic:
            return self.sample_cold(model, 
                                    batch_size, 
                                    x0, 
                                    generate=generate, 
                                    prior=prior, 
                                    t_inject=t_inject,
                                    t_diff=t_diff)
        else:
            return self.sample_ddpm(model, batch_size)
        
        


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
    




