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

from scripts.scheduler import Scheduler
from scripts.degradation import Degradation, DCTBlur, DenoisingCoefs
import scripts.dct_blur as torch_dct
from scripts.ema import ExponentialMovingAverage

    


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

        """
        Function to perform a training iteration for an arbitrarily defined Diffusion Model.

        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :param int t2: The second time step for Variable Timestep Diffusion
        :param torch.Tensor noise: The noise to degrade with for Denoising Diffusion

        :return torch.Tensor: The loss for the training iteration
        """

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

        """
        Function to perform a training epoch for an arbitrarily defined Diffusion Model.

        :param DataLoader dataloader: The dataloader containing the data
        :param bool val: Whether the model is in evaluation mode

        :return float: The average loss for the training epoch
        """
        
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
        """
        Function to sample from a trained DDPM model. Follows a common implementation of the DDPM sampling algorithm in 
        https://github.com/lucidrains/denoising-diffusion-pytorch

        :param nn.Module model: The trained DDPM model
        :param int batch_size: The batch size of the samples

        :return list: A list of samples
        """

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
    def sample_cold(self, model, batch_size = 16, x0=None, generate=False, prior=None, t2=None, t_diff=1):

        """
        Function to sample from a trained deterministic diffusion model. 
        Includes 
        - sampling for variable timestep diffusion
        - sampling a model with x0 parameterization
        - sampling a model with xt parameterization

        :param nn.Module model: The trained DDPM model
        :param int batch_size: The batch size of the samples
        :param torch.Tensor x0: The original image
        :param bool generate: Whether to generate a new x_T or degrade an existing x0 to x_T
        :param torch.Tensor prior: The prior for the VAE model
        :param int t2: The second time step for Variable Timestep Diffusion
        :param int t_diff: The difference between timesteps for Variable Timestep Diffusion

        :return list: A list of samples
        """


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


    def sample(self, model, batch_size, x0=None, prior=None, generate=False, t_diff=1):
        """
        Function to sample from a trained diffusion model.

        :param nn.Module model: The trained diffusion model
        :param int batch_size: The batch size of the samples
        :param torch.Tensor x0: The original image
        :param torch.Tensor prior: The prior for the VAE model
        :param bool generate: Whether to generate a new x_T or degrade an existing x0 to x_T
        :param int t_diff: The difference between timesteps for Variable Timestep Diffusion

        :return list: A list of samples
        """
        
        if self.deterministic:
            return self.sample_cold(model, 
                                    batch_size, 
                                    x0, 
                                    generate=generate, 
                                    prior=prior, 
                                    t_diff=t_diff)
        else:
            return self.sample_ddpm(model, batch_size)
        
        







