import argparse
import numpy as np
import os
import sys
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid

from unet import UNet
from mnist_unet import MNISTUnet
#from scripts.karras_unet import KarrasUnet
from scripts.bansal_unet import BansalUnet
from scripts.vae_unet import VAEUNet
from scripts.risannen_unet import RisannenUnet
from scripts.risannen_unet_vae import VAEUnet
from diffusion_utils import Degradation, Trainer, Sampler, ExponentialMovingAverage, DCTBlurSampling
from utils import load_dataset, create_dirs, save_video, save_gif, MyCelebA


import ipywidgets as widgets
from IPython.display import display

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']



@torch.no_grad()
def sample_func(**kwargs):

    trainloader, valloader = load_dataset(kwargs['batch_size'], kwargs['dataset'])

    x, _ = next(iter(trainloader))   
    channels = x[0].shape[0]

    # Model Configuration
    if 'mnist' in kwargs['dataset']:
        attention_levels = (2,)
        ch_mult = (1,2,2)
        num_res_blocks = 2
    elif kwargs['dataset'] == 'cifar10':
        attention_levels = (2,3)
        ch_mult = (1, 2, 2, 2)
        num_res_blocks = 4
    elif kwargs['dataset'] == 'afhq':
        attention_levels = (2,3)
        ch_mult = (1, 2, 3, 4)
        num_res_blocks = 2
    elif kwargs['dataset'] == 'celeba':
        attention_levels = (2,3)
        ch_mult = (1, 2, 2, 2)
    elif kwargs['dataset'] == 'lsun_churches':
        attention_levels = (2,3,4)
        ch_mult = (1, 2, 3, 4, 5)
        num_res_blocks = 4


    # Define Model
    if kwargs['vae']:


        # Risannen Version
        unet = VAEUnet(image_size=kwargs["image_size"],
                        in_channels=channels,
                        dim=kwargs['dim'],
                        num_res_blocks=num_res_blocks,
                        attention_levels=attention_levels,
                        dropout=0,
                        ch_mult=ch_mult,
                        latent_dim = kwargs['latent_dim'],
                        noise_scale= kwargs['noise_scale'],
                        var_timestep=True if kwargs['prediction'] == 'vxt' else False,
                        vae_loc = kwargs['vae_loc'],
                        vae_inject = kwargs['vae_inject'],
                        xt_dropout = kwargs['xt_dropout'])

    else:

        unet = RisannenUnet(image_size=kwargs["image_size"],
                            in_channels=channels,
                            dim=kwargs['dim'],
                            num_res_blocks=num_res_blocks,
                            attention_levels=attention_levels,
                            dropout=0,
                            ch_mult=ch_mult,
                            t2=True if kwargs['prediction'] == 'vxt' else False)


    
    vae_flag = "_vae" if kwargs["vae"] else ""

    # Define Trainer and Sampler
    if 'trainer' in kwargs:
        trainer = kwargs['trainer']
        kwargs.pop('trainer')
        epoch = kwargs['e']
        print("Using Trainer from kwargs")
    else:
        trainer = Trainer(model = unet, **kwargs)

        ema_flag = '' if kwargs['skip_ema'] else '_ema'

        print("Starting Sampling")
        test_string = '' if kwargs['cluster'] else 'models_to_test/'
        modelpath = f'./models/{test_string}{kwargs["dataset"]}_{kwargs["degradation"]}{vae_flag}'

        # Load Checkpoint
        try:
            chkpt = torch.load(os.path.join(f"./models/models_to_test/{kwargs['dataset']}_black_blur", f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}{manual_extension}.pt"), map_location=kwargs['device'])
            trainer.model.load_state_dict(chkpt['model_state_dict'])
            trainer.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            trainer.model_ema.load_state_dict(chkpt['ema_state_dict'])
                
            # Replace model params with EMA params 
            trainer.model_ema.copy_to(trainer.model.parameters()) # Copy EMA params to model
            
            print("Checkpoint loaded, model trained until epoch", chkpt['epoch'])
            epoch = chkpt['epoch']

        except Exception as e:
            raise ValueError("No checkpoint found, please choose pretrained variable timestep model to control VAE injections.")

    # EMA Transfer
    trainer.model.eval()
    trainer.model_ema.copy_to(trainer.model.parameters()) # Copy EMA params to model for inference
    print("EMA transferred to model")

    if kwargs['add_noise']:
        style_sampler = VAENoiseSampler(trainer, **kwargs)
    else:
        print("Sampling with VAE Noise Injections")
        style_sampler = VAESampler(trainer, **kwargs)


    img_path = os.path.join(f'./imgs/injection_imgs/')


    gridlist = []

    for step_size in kwargs['sampling_steps']:
        
        plot_grid = torch.Tensor(style_sampler.xT[0:5]).unsqueeze(0)

        # Define Injection Windows
        wb = kwargs['window_borders']
        windows = [[wb[0], wb[-1]], [wb[-1], wb[0]]] # Add pure source and target first
        intermediate_windows = [[wb[i], wb[i+1]] for i in range(len(wb)-1)] # 
        for im in intermediate_windows:
            windows.append(im)

        for inject_interval in windows:
            
            samples, xt = style_sampler.full_loop(step_size, 
                                                injection_t = inject_interval, 
                                                x0 = None,
                                                x02 = None,
                                                plot = False, 
                                                bansal_sampling = True,
                                                **kwargs)
            
            plot_grid = torch.cat((plot_grid, xt[:5].unsqueeze(0)), dim=0)

        plot_grid = torch.transpose(plot_grid, 0, 1) # Transpose to switch cols and rows in the grid
        #if not use_x:
        plot_grid = plot_grid[:,1:,:,:,:] # Remove first xT
        plot_grid = plot_grid.reshape(-1, channels, kwargs['image_size'], kwargs['image_size'])

        gridlist.append(plot_grid)

        plt.figure(figsize=(20, 10))
        grid = make_grid(plot_grid, nrow=len(windows), padding=0)
        grid = torch.clamp(grid, 0, 1)

        if not kwargs['cluster']:
            plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
            plt.axis('off')
            plt.title(f"VCD with different injection intervals \n")
            plt.show()

        dir = os.path.join(img_path, f"{wb[1:-1]}/")

        if not os.path.exists(dir):
            os.makedirs(dir)

        save_image(grid, os.path.join(dir, f"sample_injection_{step_size}.png"))



class VAESampler():

    def __init__(self, trainer, **kwargs):
        self.model = trainer.model
        self.trainer = trainer
        self.timesteps = kwargs['timesteps']
        self.T = torch.full((kwargs['n_samples'],), self.timesteps-1, dtype=torch.long).to(kwargs['device'])
        self.xT = torch.zeros((kwargs['n_samples'], kwargs['channels'], kwargs['image_size'], kwargs['image_size']), device=kwargs['device']) 
        self.target_prior = torch.randn(kwargs['n_samples'], kwargs['latent_dim']).to(kwargs['device'])
        self.target_prior = self.target_prior[0, :].unsqueeze(0).repeat(kwargs['n_samples'], 1)
        self.source_prior = torch.randn(kwargs['n_samples'], kwargs['latent_dim']).to(kwargs['device'])
        self.dct = DCTBlurSampling(trainer.degradation.dct_blur.blur_sigmas, kwargs['image_size'], kwargs['device'])

    
    @torch.no_grad()
    def full_loop(self, step_size, injection_t = [200, 50], x0 = None, x02 = None, plot = True, bansal_sampling = True, **kwargs):
        
        xt = self.xT
        samples = torch.Tensor(xt[0:5])
        
        # Bansal-style sampling
        if bansal_sampling:
            skip_steps = 1
            bansal_flag = "Bansal"
        else:
            skip_steps = step_size-1 if step_size == self.timesteps else step_size
            bansal_flag = "Regular"
            
        # Loop differs from the sequential one, as we go back to t-1 at every step
        for i in tqdm(range(self.timesteps-1, -1, -skip_steps), total = self.timesteps // skip_steps, desc = f'Sampling {bansal_flag} with step size of {step_size}'):
            
            t = torch.full((kwargs['n_samples'],), i, dtype=torch.long).to(kwargs['device'])
            t2 = torch.full((kwargs['n_samples'],), i-step_size, dtype=torch.long).to(kwargs['device'])
            
            if t2[0].item() < 0: # Equals to x0 prediction
                t2 = torch.full((kwargs['n_samples'],), -1, dtype=torch.long).to(kwargs['device']) # t-1 to account for 0 indexing that the model is seeing during training

            # if not bansal_sampling:
            #     print(f"t = {t[0].item()}, t2 = {t2[0].item()}")
            
            # Inject source information in a certain t range
            # if x0 and / or x02 are not None, take the conditioning information from the encoded data,
            if i < injection_t[0] and i > injection_t[1]:
                if x02 is not None:
                    xt2 = self.trainer.degradation.degrade(x02, t)
                    cond2 = self.trainer.degradation.degrade(x02, t2)
                    dummy_pred = self.model(xt2, t, cond=cond2, prior=None, t2=t2).detach() # Dummy prediction to get the encoded VAE latent
                    prior = self.model.vae_noise.detach() 
                else:
                    prior = self.source_prior 
                cond = None # No condition for the model, such that is has to use the prior
            else:
                if x0 is not None:
                    cond = self.trainer.degradation.degrade(x0, t2)
                    prior = None # No prior for the model, such that it has to use the condition
                else:
                    cond = None
                    prior = self.target_prior

            pred = self.model(xt, t, cond=cond, prior=prior, t2=t2)
            pred = pred.detach()
            
            xtm1_model = xt + pred
            samples = torch.cat((samples, xtm1_model[0:5]), dim=0)

            # Bansal Part
            if bansal_sampling:
                if step_size == self.timesteps:
                    
                    xt_hat = self.trainer.degradation.degrade(xtm1_model, t)
                    xtm1_hat = self.trainer.degradation.degrade(xtm1_model, t-1)
                    
                else:

                    # Calculcate reblurring for DCT
                    stdt = self.trainer.degradation.blur.dct_sigmas[t[0].item()]
                    stdtm1 = self.trainer.degradation.blur.dct_sigmas[t[0].item()-1]
                    stdt2 = self.trainer.degradation.blur.dct_sigmas[t2[0].item()] if t2[0].item() != -1 else 0

                    # Manually calculating DCT timesteps
                    t_dct = torch.full((kwargs['n_samples'],), stdt**2/2, dtype=torch.float32)[:, None, None, None].to(kwargs['device'])
                    tm1_dct = torch.full((kwargs['n_samples'],), stdtm1**2/2, dtype=torch.float32)[:, None, None, None].to(kwargs['device'])
                    t2_dct = torch.full((kwargs['n_samples'],), stdt2**2/2, dtype=torch.float32)[:, None, None, None].to(kwargs['device'])
                    
                    # Calculate DCT differences to get to t / t-1 from t2
                    t_diff_dct = t_dct - t2_dct
                    tm1_diff_dct = tm1_dct - t2_dct

                    # Undo Blacking
                    blacking_coef = self.trainer.degradation.blacking_coefs[t2[0].item()] if t2[0].item() != -1 else 1
                    xtm1_model = xtm1_model / blacking_coef

                    # Reblur to t / t-1 from t2
                    xt_hat = self.dct(xtm1_model, t=t_diff_dct) 
                    xtm1_hat = self.dct(xtm1_model, t=tm1_diff_dct)

                    # Reapply Blacking
                    xt_hat = xt_hat * self.trainer.degradation.blacking_coefs[t[0].item()]
                    xtm1_hat = xtm1_hat * self.trainer.degradation.blacking_coefs[t[0].item() - 1]

                    # Check for NAs
                    assert not torch.isnan(xtm1_hat).any() 

                    if t[0].item()-1 == -1:
                        xtm1_hat = xtm1_model # Keep the original image for t=-1 (needed for Bansal style sampling)
                        #assert torch.equal(xtm1_hat[0], xtm1_model[0]), f"DCT reblurring failed, xtm1_hat is not equal to xtm1_model with xtm1_hat = {xtm1_hat[0,0,0]} and xtm1_model = {xtm1_model[0,0,0]}"

                xt = xt - xt_hat + xtm1_hat # Counter the bias of the model prediction by having it incorporated two times in the sampling process
                samples = torch.cat((samples, xt[0:5]), dim=0)

            else:
                xt = xtm1_model
                #samples = torch.cat((samples, xt[0:5]), dim=0)
                if t[0].item() < step_size:
                    break
        
        if plot:
            self.plot(xt)
            save_gif(samples[::5,:,:], f'./imgs/experiment_imgs', nrow = 1, name = f"sample_DCT_{bansal_flag}_{step_size}.gif")

        return samples, xt



    def plot(self, input):
        plt.figure(figsize=(20, 10))
        grid = make_grid(input, nrow=10, padding=0)
        grid = torch.clamp(grid, 0, 1)
        plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.show()



class VAENoiseSampler():

    def __init__(self, trainer, **kwargs):
        self.model = trainer.model
        self.trainer = trainer
        self.timesteps = kwargs['timesteps']
        self.T = torch.full((kwargs['n_samples'],), self.timesteps-1, dtype=torch.long).to(kwargs['device'])
        self.xT = torch.zeros((kwargs['n_samples'], kwargs['channels'], kwargs['image_size'], kwargs['image_size']), device=kwargs['device']) 
        self.target_prior = torch.randn(kwargs['n_samples'], kwargs['latent_dim']).to(kwargs['device'])
        self.target_prior = self.target_prior[0, :].unsqueeze(0).repeat(kwargs['n_samples'], 1)
        self.source_prior = torch.randn(kwargs['n_samples'], kwargs['latent_dim']).to(kwargs['device'])
        self.sample_noise = torch.randn(kwargs['n_samples'], self.model.in_channels, kwargs['image_size'], kwargs['image_size']).to(kwargs['device'])  
        self.dct = DCTBlurSampling(trainer.degradation.dct_blur.blur_sigmas, kwargs['image_size'], kwargs['device'])

    
    @torch.no_grad()
    def full_loop(self, step_size, injection_t = [200, 50], x0 = None, x02 = None, plot = True, bansal_sampling = True, **kwargs):
        
        self.model.add_noise = True
        self.model.eval()
        
        xt = self.xT
        samples = torch.Tensor(xt[0:5])
        
        if step_size == self.timesteps:
            skip_steps = 1
        else:
            skip_steps = step_size-1 if step_size == self.timesteps else step_size

        # Loop differs from the sequential one, as we go back to t-1 at every step
        for i in tqdm(range(self.timesteps-1, -1, -skip_steps), total = self.timesteps // 1):
            
            t = torch.full((kwargs['n_samples'],), i, dtype=torch.long).to(kwargs['device'])
            t2 = torch.full((kwargs['n_samples'],), i-step_size, dtype=torch.long).to(kwargs['device'])
            
            if t2[0].item() < 0: # Equals to x0 prediction
                t2 = torch.full((kwargs['n_samples'],), -1, dtype=torch.long).to(kwargs['device']) # t-1 to account for 0 indexing that the model is seeing during training
            
            t2 = None
            
            # Uncomment to sample new noise every time
            # self.model.sample_noise = None

            # # Set Sample Noise
            # if self.model.sample_noise is None:
            #     self.model.sample_noise = self.sample_noise 

            if i < injection_t[0] and i > injection_t[1]:
                if x02 is not None:
                    xt2 = self.trainer.degradation.degrade(x02, t)
                    cond2 = self.trainer.degradation.degrade(x02, t2)
                    _ = self.model(xt2, t, cond=cond2, prior=None, t2=t2).detach() # Dummy prediction to get the encoded VAE latent
                    prior = self.model.vae_noise.detach() 
                else:
                    prior = self.source_prior 
                cond = None # No condition for the model, such that is has to use the prior
            else:
                if x0 is not None:
                    cond = self.trainer.degradation.degrade(x0, t2)
                    prior = None # No prior for the model, such that it has to use the condition
                else:
                    cond = None
                    prior = self.target_prior
                
            pred = self.model(xt, t, cond=cond, prior=prior, t2=t2)
            pred = pred.detach()
            
            xtm1_model = pred #xt + pred
            samples = torch.cat((samples, xtm1_model[0:5]), dim=0)

            # Bansal Part
            if bansal_sampling:
                    
                xt_hat = self.trainer.degradation.degrade(xtm1_model, t)
                xtm1_hat = self.trainer.degradation.degrade(xtm1_model, t - 1) # This returns x0_hat for t=0
                xt = xt - xt_hat + xtm1_hat
                
                samples = torch.cat((samples, xt[0:5]), dim=0)

            else:
                xt = xtm1_model
                #samples = torch.cat((samples, xt[0:5]), dim=0)
                if t[0].item() < step_size:
                    break
        
        if plot:
            self.plot(xt)
            save_gif(samples[::5,:,:], f'./imgs/experiment_imgs', nrow = 1, name = f"sample_DCT_{bansal_flag}_{step_size}.gif")

        return samples, xt
    



    def plot(self, input):
        plt.figure(figsize=(20, 10))
        grid = make_grid(input, nrow=10, padding=0)
        grid = torch.clamp(grid, 0, 1)
        plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.show()




if __name__ == "__main__":
    
    load = "vae"

    if load == "vae":
        prediction = "vxt"
        degradation = "black_blur"
        add_noise = 'store_true'
        vae = 'store_false'
        manual_extension = '_emb'#'_add_best'

    elif load == "risannen":
        prediction = "xt"
        degradation = "black_blur"
        add_noise = 'store_false'
        vae = 'store_true'
        manual_extension = '_risannen'

    elif load == "bansal":
        prediction = "residual"
        degradation = "noise"
        add_noise = 'store_true'
        vae = 'store_true'
        manual_extension = '_ddpm'



    parser = argparse.ArgumentParser(description='Diffusion Models')

    # General Diffusion Parameters
    parser.add_argument('--timesteps', '--t', type=int, default=200, help='Degradation timesteps')
    parser.add_argument('--prediction', '--pred', type=str, default=prediction, help='Prediction method, choose one of [x0, xt, residual]')
    parser.add_argument('--dataset', type=str, default='afhq', help='Dataset to run Diffusion on. Choose one of [mnist, cifar10, celeba, lsun_churches]')
    parser.add_argument('--degradation', '--deg', type=str, default=degradation, help='Degradation method')
    parser.add_argument('--batch_size', '--b', type=int, default=32, help='Batch size')
    parser.add_argument('--dim', '--d', type=int , default=128, help='Model dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', '--e', type=int, default=20, help='Number of Training Epochs')
    parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
    parser.add_argument('--loss_weighting', action='store_false', help='Whether to use weighting for reconstruction loss')
    parser.add_argument('--var_sampling_step', type=int, default = 1, help='How to sample var timestep model - int > 0 indicates t difference to predict, -1 indicates x0 prediction')
    parser.add_argument('--min_t2_step', type=int, default=1, help='With what min step size to discretize t2 in variational timestep model') 
    parser.add_argument('--baseline', '--base', type=str, default='xxx', help='Whether to run a baseline model - Risannen, Bansal, VAE')
    parser.add_argument('--sampling_steps', type=list, default=[200, 100], help='Which steps to sample at')
    parser.add_argument('--window_borders', type=list, default=[200, 150, 100, 50, -1], help='When to cut off injection windows')
    
    # Noise Injection Parameters
    parser.add_argument('--vae', action=vae, help='Whether to use VAE Noise injections')
    parser.add_argument('--vae_alpha', type=float, default = 0.999, help='Trade-off parameter for weight of Reconstruction and KL Div')
    parser.add_argument('--latent_dim', type=int, default=128, help='Which dimension the VAE latent space is supposed to have')
    parser.add_argument('--ris_noise', action=add_noise, help='Whether to add noise Risannen et al. style')
    parser.add_argument('--add_noise', action=add_noise, help='Whether to add noise at any other location')
    parser.add_argument('--break_symmetry', action='store_true', help='Whether to add noise to xT Bansal et al. style')
    parser.add_argument('--noise_scale', type=float, default = 0.01, help='How much Noise to add to the input')
    parser.add_argument('--vae_loc', type=str, default = 'emb', help='Where to inject VAE Noise. One of [start, bottleneck, emb].')
    parser.add_argument('--vae_inject', type=str, default = 'concat', help='How to inject VAE Noise. One of [concat, add].')
    parser.add_argument('--xt_dropout', type=float, default = 0.2, help='How much of xt is dropped out at every step (to foster reliance on VAE injections)')

    # Housekeeping Parameters
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to try to load a checkpoint')
    parser.add_argument('--sample_interval', type=int, help='After how many epochs to sample', default=1)
    parser.add_argument('--n_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--fix_sample', action='store_false', help='Whether to fix x_T for sampling, to see sample progression')
    parser.add_argument('--skip_ema', action='store_true', help='Whether to skip model EMA')
    parser.add_argument('--model_ema_decay', type=float, default=0.997, help='Model EMA decay')
    parser.add_argument('--cluster', action='store_true', help='Whether to run script locally')
    parser.add_argument('--skip_wandb', action='store_true', help='Whether to skip wandb logging')
    parser.add_argument('--verbose', '--v', action='store_true', help='Verbose mode')

    parser.add_argument('--test_run', action='store_true', help='Whether to test run the pipeline')

    args = parser.parse_args()

    args.num_downsamples = 2 if args.dataset == 'mnist' else 3
    args.device = 'cuda' if torch.cuda.is_available() else 'mps'

    if 'mnist' in args.dataset:
        args.image_size = 28
        args.channels = 1
    elif args.dataset == 'cifar10':
        args.image_size = 32
        args.channels = 3
    elif args.dataset == 'afhq':
        args.image_size = 64
        args.channels = 3
    

    if args.vae:
        print("Using VAE Noise Injections")
        assert not args.add_noise, "Cannot use VAE and add noise at the same time"
    else:
        if args.add_noise:
            print("Using Risannen Noise Injections")
        else:
            print("Using Normal U-Net")

    if not args.cluster:
        print("Running locally, Cluster =", args.cluster)
        if args.device == 'cuda':
            warnings.warn('Consider running model on cluster-scale if CUDA is available')

    if args.test_run:
        print("Running Test Run with only one iter per epoch")

    print("Device: ", args.device)


    # Run main function
    sample_func(**vars(args))


    print("Finished Training")









