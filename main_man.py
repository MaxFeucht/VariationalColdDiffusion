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
from cleanfid import fid

from scripts.nets.unet import Unet
from scripts.diffusion import Degradation, Trainer, Sampler, ExponentialMovingAverage
from scripts.utils import load_dataset, plot_degradation, create_dirs, save_video, save_single_imgs, save_gif, MyCelebA

from sampler import sample_func

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']


def main(**kwargs):
    
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

    kwargs['ch_mult'] = ch_mult

    # Select right VAE Unet
    if kwargs['multiscale_vae']:
        from scripts.nets.multiscale_vae import VAEUnet
    else:
        from scripts.nets.vae_unet import VAEUnet

    # Define Model
    if kwargs['vae']:

        print(kwargs['dim'])

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

    if kwargs['autoencoder']:

        from scripts.nets.conditional_unet import AEUnet

        unet = AEUnet(image_size=kwargs["image_size"],
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


    # Print Number of Parameters
    print(f"Number of Parameters: {sum(p.numel() for p in unet.parameters())}")

    # Print Setup
    print("Setup: \n\n ", kwargs)

    # Enable Multi-GPU training
    if kwargs['multi_gpu']:
        print("Using", torch.cuda.device_count(), "GPUs")
        unet = nn.DataParallel(unet)

    # Define Trainer and Sampler
    trainer = Trainer(model = unet, **kwargs)
    sampler = Sampler(**kwargs)
    
    # Fit GMM for cold sampling in deblurring diffusion
    #if kwargs['degradation'] in ['blur', 'pixelation']:
    sampler.fit_gmm(trainloader, clusters=1)

    # Fix x_T for sampling
    if kwargs['fix_sample']:
        sampler.sample_x_T(kwargs['n_samples'], channels, kwargs["image_size"])

    # Fix Prior for VAE
    if not kwargs['vae_loc'] == 'bold' and not kwargs['multiscale_vae']:
        prior = torch.randn((kwargs['n_samples'], kwargs['latent_dim'])).to(kwargs['device'])        
    elif kwargs['vae_loc'] == 'bold':
        prior = torch.randn((kwargs['n_samples'], channels, kwargs['image_size'], kwargs['image_size'])).to(kwargs['device'])
    elif kwargs['multiscale_vae']:
        prior = torch.randn((kwargs['n_samples'], len(ch_mult), kwargs['latent_dim'])).to(kwargs['device'])

    # Fix x0 for conditional generation - allows to disseminate the effect of the encoder vs the contribution of noise
    condition_x0 = next(iter(trainloader))[0][:kwargs['n_samples']].to(kwargs['device']) 

    # Create directories
    imgpath, modelpath = create_dirs(**kwargs)
    ema_flag = '' if kwargs['skip_ema'] else '_ema'


    if kwargs['fid_only']:
        print("Skipping training to only calculating FID Score")
        kwargs['load_checkpoint'] = True
        kwargs['epochs'] = 0
    

    # Load Checkpoint
    if kwargs['load_checkpoint']:
        try:
            chkpt = torch.load(os.path.join(modelpath, f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}.pt"), map_location=kwargs['device'])
            chkpt['model_state_dict'] = {k.replace('module.', ''): v for k, v in chkpt['model_state_dict'].items()}
            chkpt['optimizer_state_dict'] = {k.replace('module.', ''): v for k, v in chkpt['optimizer_state_dict'].items()}
            chkpt['ema_state_dict'] = {k.replace('module.', ''): v for k, v in chkpt['ema_state_dict'].items()}
            trainer.model.load_state_dict(chkpt['model_state_dict'])
            trainer.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            trainer.model_ema.load_state_dict(chkpt['ema_state_dict'])
            epoch_offset = chkpt['epoch']
            trainer.annealing_factor = epoch_offset * kwargs['batch_size']
            print("Checkpoint loaded, continuing training from epoch", epoch_offset)
        except Exception as e:
            print("No checkpoint found: ", e)
            epoch_offset = 0
    else:
        epoch_offset = 0



    # Training Loop
    for e in range(epoch_offset + 1, kwargs['epochs']):
        
        sample_flag = True if (e) % kwargs['sample_interval'] == 0 else False 

        # Train
        trainer.model.train()
        if kwargs['vae'] and not kwargs['autoencoder']:
            trainloss, reconstruction, kl_div = trainer.train_epoch(trainloader, val=False) # ATTENTION: CURRENTLY NO VALIDATION LOSS
            if not kwargs['skip_wandb']:
                wandb.log({"train loss": trainloss,
                        "reconstruction loss": reconstruction,
                            "kl divergence": kl_div}, step = e)
            print(f"Epoch {e} Train Loss: {trainloss}, \nReconstruction Loss: {reconstruction}, \nKL Divergence: {kl_div}")
            if kwargs['multiscale_vae']:
                print(f"Single KL Divergences of last batch: {trainer.model.kls}")
        else:
            trainloss = trainer.train_epoch(trainloader, val=False)
            if not kwargs['skip_wandb']:
                wandb.log({"train loss": trainloss}, step=e)
            print(f"Epoch {e} Train Loss: {trainloss}")

        if sample_flag:

            # Validation
            
            # Sample from model using EMA parameters
            trainer.model.eval()
            trainer.model_ema.store(trainer.model.parameters()) # Store model params
            trainer.model_ema.copy_to(trainer.model.parameters()) # Copy EMA params to model

            # Sample
            nrow = 6

            if kwargs['degradation'] in ['noise', 'fadeblack_noise'] : # Noise Sampling
                
                samples, xt = sampler.sample(trainer.model, kwargs['n_samples'])

                save_image(samples[-1], os.path.join(imgpath, f'sample_{e}.png'), nrow=nrow) #int(math.sqrt(kwargs['n_samples']))
                save_video(samples, imgpath, nrow, f'sample_{e}.mp4')
            
            else: 
                
                if not kwargs['autoencoder']:
                    # Cold Sampling
                    # Conditional Sampling
                    t_diff_var = kwargs['var_sampling_step'] #if e % 2 != 0 else -1 # Alternate between sampling xt-t_diff style and x0 style
                    t_diff_xt = kwargs['var_sampling_step'] if kwargs['prediction'] == 'xt' else 1 # Sampling xt-1 style but with bigger steps 
                    _, _, _, all_images = sampler.sample(model=trainer.model, 
                                                                            generate=False, 
                                                                            x0=condition_x0,
                                                                            batch_size = kwargs['n_samples'],
                                                                            t_diff=t_diff_var if kwargs['prediction'] == 'vxt' else t_diff_xt) # Sample xt-1 style every second epoch
                    
                    save_image(all_images, os.path.join(imgpath, f'cond_{e}.png'), nrow=nrow)

                    # Unconditional Sampling
                    # Prior is defined above under "fix_sample"
                    gen_samples, _, _, gen_all_images = sampler.sample(model = trainer.model,
                                                                            generate=True,
                                                                            batch_size = kwargs['n_samples'], 
                                                                            prior=prior,
                                                                            t_diff=t_diff_var if kwargs['prediction'] == 'vxt' else t_diff_xt) # Sample xt-1 style every second epoch
                
                    save_image(gen_all_images, os.path.join(imgpath, f'prior_{e}.png'), nrow=nrow)

                    if kwargs['cold_perturb']:
                        save_gif(gen_samples, imgpath, nrow, f'sample_{e}.gif')

                else:
                    # Conditional Sampling only kind of sampling possible with autoencoder
                    #og_img = next(iter(trainloader))[0][:kwargs['n_samples']].to(kwargs['device'])
                    t_diff_var = kwargs['var_sampling_step'] #if e % 2 != 0 else -1 # Alternate between sampling xt-t_diff style and x0 style
                    t_diff_xt = kwargs['var_sampling_step'] if kwargs['prediction'] == 'xt' else 1 # Sampling xt-1 style but with bigger steps 
                    _, _, _, all_images = sampler.sample(model=trainer.model, 
                                                                            generate=False, 
                                                                            x0=condition_x0,
                                                                            batch_size = kwargs['n_samples'],
                                                                            t_diff=t_diff_var if kwargs['prediction'] == 'vxt' else t_diff_xt) # Sample xt-1 style every second epoch

                    save_image(all_images, os.path.join(imgpath, f'ae_cond_{e}.png'), nrow=nrow)

                # After sampling, restore model parameters
                trainer.model_ema.restore(trainer.model.parameters()) # Restore model params

            # save_gif(samples, imgpath, nrow, f'sample_{e}.gif')

            # Save checkpoint
            if not kwargs['test_run']:
                chkpt = {
                    'epoch': e,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'ema_state_dict': trainer.model_ema.state_dict(),
                    'kwargs': kwargs,
                }
                torch.save(chkpt, os.path.join(modelpath, f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}.pt"))
        
        # # Sample Function for VXT
        # if kwargs['prediction'] == 'vxt' and not kwargs['autoencoder']:
        #     if e % 20 == 0: 
        #         sample_args = kwargs.copy()
        #         sample_args['trainer'] = trainer
        #         sample_args['cluster'] = True
        #         sample_args['sampling_steps'] = [kwargs['min_t2_step']]
        #         sample_args['e'] = e
        #         sample_func(**sample_args)

        
    ## Calculate FID
    print("Calculating FID Score")

    # Save Diffusion Samples for FID calculation
    fid_path = imgpath.split('run_')[0] + 'fid/' + kwargs['prediction'] + '_' + str(kwargs['timesteps'])
    if not os.path.exists(fid_path):
        print("Creating FID Path")
        os.makedirs(fid_path)

        # Sample from model using EMA parameters
        trainer.model.eval()
        trainer.model_ema.store(trainer.model.parameters()) # Store model params
        trainer.model_ema.copy_to(trainer.model.parameters()) # Copy EMA params to model

        idx = 0
        iters = len(trainloader)
        sampler.sample_x_T(kwargs['batch_size'], channels, kwargs['image_size'])
        for i in tqdm(range(iters), total = iters, desc='Generating FID Samples'):

            if kwargs['degradation'] in ['noise', 'fadeblack_noise'] : # Noise Sampling
                sample_out, xt = sampler.sample(trainer.model, kwargs['n_samples'])

            else:
                prior = torch.randn((kwargs['batch_size'], kwargs['latent_dim'])).to(kwargs['device'])
                sample_out = sampler.sample(model=trainer.model, 
                                                    generate=False if 'bansal' in kwargs['baseline'] else True, 
                                                    x0=condition_x0 if 'bansal' in kwargs['baseline'] else None,
                                                    batch_size = kwargs['batch_size'],
                                                    prior=prior,
                                                    t_diff=kwargs['var_sampling_step'] if kwargs['prediction'] == 'vxt' else 1)

            all_images = sample_out[-1]
            idx = save_single_imgs(all_images, fid_path, idx)
    else:
        print("Model FID Path already exists")
    
    # Save Dataset images for FID calculation
    if kwargs['dataset'] == 'afhq':
        dataset_imgs_pth = './data/AFHQ_64/train/'
    elif kwargs['dataset'] == 'lsun_churches':
        dataset_imgs_pth = './data/LSUN/churches/'
    elif kwargs['dataset'] == 'cifar10':
        dataset_imgs_pth = './data/cifar10_imgs/'
    elif 'mnist' in kwargs['dataset']:
        dataset_imgs_pth = './data/mnist_imgs/'

    if not os.path.exists(dataset_imgs_pth):
        os.makedirs(dataset_imgs_pth)
        idx = 0 
        for batch in tqdm(trainloader, desc='Saving Dataset Images'):
            idx = save_single_imgs(batch[0], dataset_imgs_pth, idx)
        
        print("Saved Dataset images to folder")
    

    score = fid.compute_fid(dataset_imgs_pth, fid_path, device=kwargs['device'], num_workers=0)

    print(f"FID: {score}")
    if not kwargs['skip_wandb']:
        wandb.log({"FID": score})






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


    if args.baseline == 'ddpm':
        args.prediction = 'residual'
        args.degradation = 'noise'
        args.vae = False
        args.ris_noise = False
        args.break_symmetry = False
    elif args.baseline == 'risannen':
        args.vae = False
        args.ris_noise = True
        args.break_symmetry = False
        args.degradation = 'blur'
        args.prediction = 'xt'
        args.noise_scale = 0.01
        args.timesteps = 200
        args.min_t2_step = 1
    elif args.baseline == 'bansal':
        args.vae = False
        args.ris_noise = False
        args.break_symmetry = True
        args.degradation = 'blur'
        args.prediction = 'x0'
        args.noise_scale = 0.002
    elif args.baseline == 'vae_xt':
        args.vae = True
        args.ris_noise = False
        args.break_symmetry = False
        args.prediction = 'xt'
    elif args.baseline == 'vae_x0':
        args.vae = True
        args.ris_noise = False
        args.break_symmetry = False
        args.prediction = 'x0'


    if args.prediction == 'vxt':
        var_string = "Running Variable Timestep Diffusion"
    else:
        var_string = "Running Sequential Diffusion"

    if not args.cluster:
        print("Running locally, Cluster =", args.cluster)
        if args.device == 'cuda':
            warnings.warn('Consider running model on cluster-scale if CUDA is available')
    
    if args.test_run:
        print("Running Test Run with only one iter per epoch")

    if args.vae:
        setup_string = "using VAE Noise Injections"
        # assert not args.ris_noise, "Cannot use VAE and add Risannen noise at the same time"
    else:
        if args.ris_noise:
            setup_string = "with Risannen Noise Injections"
        else:
            setup_string = "with Normal U-Net"
    
    print(var_string + " " + setup_string)
    
    # Initialize wandb
    if not args.skip_wandb:
        wandb.init(
        project="Diffusion Thesis",
        config=vars(args))
    
    print("Device: ", args.device)


    # Run main function
    main(**vars(args))

    # Finish wandb run
    if not args.test_run:
        wandb.finish()

    print("Finished Training")