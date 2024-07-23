import argparse
import numpy as np
import os
import sys
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import wandb

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torchvision.utils import save_image


from scripts.diffusion import Trainer, VarTSampler
from scripts.utils import load_dataset, plot_degradation
from scripts.nets.unet import Unet
from scripts.nets.vae_unet import VAEUnet

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']

@torch.no_grad()
def sample_func(**kwargs):
    
    trainloader, valloader = load_dataset(kwargs['batch_size'], kwargs['dataset'])
    
    if kwargs['verbose']:
        plot_degradation(train_loader=trainloader, **kwargs)
    
    x, _ = next(iter(trainloader))   
    channels = x[0].shape[0]

    # Model Configuration
    if 'mnist' in kwargs['dataset']:
        attention_levels = (2,)
        ch_mult = (1,2,2)
        num_res_blocks = 2
        dropout = 0.1
    elif kwargs['dataset'] == 'cifar10':
        attention_levels = (2,3)
        ch_mult = (1, 2, 2, 2)
        num_res_blocks = 4
        dropout = 0.1
    elif kwargs['dataset'] == 'afhq':
        attention_levels = (2,3)
        ch_mult = (1, 2, 3, 4)
        num_res_blocks = 2
        dropout = 0.2
    elif kwargs['dataset'] == 'celeba':
        attention_levels = (2,3)
        ch_mult = (1, 2, 2, 2)
    elif kwargs['dataset'] == 'lsun_churches':
        attention_levels = (2,3,4)
        ch_mult = (1, 2, 3, 4, 5)
        num_res_blocks = 4
        dropout = 0.2

    # Define Model
    if kwargs['vae']:

        # Bansal Version
        # unet = VAEUnet(image_size=imsize,
        #                 channels=channels,
        #                 out_ch=channels,
        #                 ch=kwargs['dim'],
        #                 ch_mult= ch_mult,
        #                 num_res_blocks=num_res_blocks,
        #                 attn_resolutions=(14,) if kwargs['dataset'] == 'mnist' else (16,),
        #                 latent_dim=int(channels*imsize*imsize//kwargs['vae_downsample']),
        #                 noise_scale=kwargs['noise_scale'],
        #                 dropout=0)

        # Risannen Version
        unet = VAEUnet(image_size=kwargs["image_size"],
                        in_channels=channels,
                        dim=kwargs['dim'],
                        num_res_blocks=num_res_blocks,
                        attention_levels=attention_levels,
                        dropout=dropout,
                        ch_mult=ch_mult,
                        latent_dim = kwargs['latent_dim'],
                        add_noise=kwargs['add_noise'],
                        noise_scale= kwargs['noise_scale'],
                        var_timestep=True if kwargs['prediction'] in ['xt', 'vxt'] else False,
                        vae_loc = kwargs['vae_loc'],
                        vae_inject = kwargs['vae_inject'],
                        xt_dropout = kwargs['xt_dropout'])

    else:
    
        unet = Unet(image_size=kwargs["image_size"],
                    in_channels=channels,
                    dim=kwargs['dim'],
                    num_res_blocks=num_res_blocks,
                    attention_levels=attention_levels,
                    dropout=dropout,
                    ch_mult=ch_mult,
                    add_noise=kwargs['add_noise'],
                    noise_scale= kwargs['noise_scale'],
                    t2=True if kwargs['prediction'] in ['xt', 'vxt'] else False)

    
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
            chkpt = torch.load(os.path.join(modelpath, f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}.pt"), map_location=kwargs['device'])
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

    sampler = VarTSampler(trainer, **kwargs)

    img_path = os.path.join(f'./imgs/sampled_imgs/{kwargs["dataset"]}_{kwargs["degradation"]}{vae_flag}_{kwargs["min_t2_step"]}_{kwargs["vae_inject"]}_{kwargs["vae_loc"]}/')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    nrow = 6
    for step_size in kwargs['sampling_steps']:
        
        step_size = int(step_size)

        # Regular Sampling
        samples, xt = sampler.full_loop(step_size=step_size, bansal_sampling=False)
        save_image(xt, os.path.join(img_path, f'sequential_{step_size}.png'), nrow=nrow)
        #save_gif(samples[0], img_path, nrow, f'sequential_{step_size}.gif')

        # Redegradation Sampling
        samples, xt = sampler.full_loop(step_size=step_size, bansal_sampling=True)
        save_image(xt, os.path.join(img_path, f'redegradation_{step_size}.png'), nrow=nrow)
        #save_gif(samples[0], img_path, nrow, f'redegradation_{step_size}.gif')






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Diffusion Models')

    # General Diffusion Parameters
    parser.add_argument('--timesteps', '--t', type=int, default=6, help='Degradation timesteps')
    parser.add_argument('--prediction', '--pred', type=str, default='vxt', help='Prediction method, choose one of [x0, xt, residual]')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to run Diffusion on. Choose one of [mnist, cifar10, celeba, lsun_churches]')
    parser.add_argument('--degradation', '--deg', type=str, default='black_blur', help='Degradation method')
    parser.add_argument('--batch_size', '--b', type=int, default=64, help='Batch size')
    parser.add_argument('--dim', '--d', type=int , default=64, help='Model dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', '--e', type=int, default=3, help='Number of Training Epochs')
    parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
    parser.add_argument('--loss_weighting', action='store_true', help='Whether to use weighting for reconstruction loss')
    parser.add_argument('--var_sampling_step', type=int, default = 1, help='How to sample var timestep model - int > 0 indicates t difference to predict, -1 indicates x0 prediction')
    parser.add_argument('--min_t2_step', type=int, default=1, help='With what min step size to discretize t2 in variational timestep model')
    parser.add_argument('--baseline', '--base', type=str, default='xxxx', help='Whether to run a baseline model - Risannen, Bansal, VAE')
    parser.add_argument('--multiscale_vae', action='store_true', help='Whether to use multiscale vae encoder instead of normal vae encoder')
    parser.add_argument('--autoencoder', action='store_true', help='Whether to use autoencoder instead of vae encoder')
    parser.add_argument('--kl_annealing', type=int, default = 5000, help='Number of epochs in which to anneal KL divergence')
    parser.add_argument('--cold_perturb', action='store_true', help='Whether to use cold perturbation')

    # Noise Injection Parameters
    parser.add_argument('--vae', action='store_true', help='Whether to use VAE Noise injections')
    parser.add_argument('--vae_beta', type=float, default = 0.1, help='Trade-off parameter for weight of Reconstruction and KL Div')
    parser.add_argument('--latent_dim', type=int, default=32, help='Which dimension the VAE latent space is supposed to have')
    parser.add_argument('--ris_noise', action='store_true', help='Whether to add noise Risannen et al. style')
    parser.add_argument('--add_noise', action='store_true', help='Whether to add noise somewhere in the process')
    parser.add_argument('--break_symmetry', action='store_true', help='Whether to add noise to xT Bansal et al. style')
    parser.add_argument('--noise_scale', type=float, default = 0.01, help='How much Noise to add to the input')
    parser.add_argument('--vae_loc', type=str, default = 'emb', help='Where to inject VAE Noise. One of [start, bottleneck, emb].')
    parser.add_argument('--vae_inject', type=str, default = 'concat', help='How to inject VAE Noise. One of [concat, add].')
    parser.add_argument('--xt_dropout', type=float, default = 0, help='How much of xt is dropped out at every step (to foster reliance on VAE injections)')

    # Housekeeping Parameters
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to try to load a checkpoint')
    parser.add_argument('--sample_interval', type=int, help='After how many epochs to sample', default=1)
    parser.add_argument('--n_samples', type=int, default=60, help='Number of samples to generate')
    parser.add_argument('--fix_sample', action='store_false', help='Whether to fix x_T for sampling, to see sample progression')
    parser.add_argument('--skip_ema', action='store_true', help='Whether to skip model EMA')
    parser.add_argument('--model_ema_decay', type=float, default=0.998, help='Model EMA decay')
    parser.add_argument('--cluster', action='store_true', help='Whether to run script locally')
    parser.add_argument('--skip_wandb', action='store_true', help='Whether to skip wandb logging')
    parser.add_argument('--verbose', '--v', action='store_true', help='Verbose mode')
    parser.add_argument('--multi_gpu', action='store_true', help='Whether to use multi-gpu training')
    parser.add_argument('--fid_only', action='store_true', help='Whether to calculate only the FID score')

    parser.add_argument('--test_run', action='store_true', help='Whether to test run the pipeline')

    args = parser.parse_args()

    args.num_downsamples = 2 if args.dataset == 'mnist' else 3
    args.device = 'cuda' if torch.cuda.is_available() else 'mps'

    if 'mnist' in args.dataset:
        args.image_size = 28
    elif args.dataset == 'cifar10':
        args.image_size = 32
    elif args.dataset == 'afhq':
        args.image_size = 64
    elif args.dataset == 'lsun_churches':
        args.image_size = 256

    if args.prediction == 'vxt':
        var_string = "Running Variable Timestep Diffusion"
    else:
        var_string = "Running Sequential Diffusion"

    if args.vae:
        setup_string = "using VAE Noise Injections"
        assert not args.add_noise, "Cannot use VAE and add noise at the same time"
    else:
        if args.add_noise:
            setup_string = "with Risannen Noise Injections"
        else:
            setup_string = "with Normal U-Net"
    
    print(var_string + " " + setup_string)


    if not args.cluster:
        print("Running locally, Cluster =", args.cluster)
        # args.dim = int(args.dim/2)
        if args.device == 'cuda':
            warnings.warn('Consider running model on cluster-scale if CUDA is available')
    
    if args.test_run:
        print("Running Test Run with only one iter per epoch")

    if args.baseline == 'risannen':
        args.vae = False
        args.add_noise = True
        args.break_symmetry = False
        args.prediction = 'xt'
        args.noise_scale = 0.01
    elif args.baseline == 'bansal':
        args.vae = False
        args.add_noise = False
        args.break_symmetry = True
        args.prediction = 'x0'
        args.noise_scale = 0.002
    elif args.baseline == 'vae_xt':
        args.vae = True
        args.add_noise = False
        args.break_symmetry = False
        args.prediction = 'xt'
    elif args.baseline == 'vae_x0':
        args.vae = True
        args.add_noise = False
        args.break_symmetry = False
        args.prediction = 'x0'

    print("Device: ", args.device)

    # Run main function
    sample_func(**vars(args))


    print("Finished Training")









