import os 
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision import datasets
from torchvision.datasets import CelebA
from torchvision import transforms as T

from scripts.diffusion import Degradation, Trainer, Sampler, ExponentialMovingAverage
from scripts.data.datasets import load_data


def create_dirs(**kwargs):

    vae_flag = "_vae" if kwargs["vae"] else ""
    noise_flag = "_noise" if kwargs["add_noise"] else ""
    vae_inject_flag = "_" + kwargs["vae_inject"] if kwargs["vae"] else ""

    # Check if directory for imgs exists
    imgpath = f'./imgs/{kwargs["dataset"]}_{kwargs["degradation"]}{vae_flag}{noise_flag}'
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)
    dirs = os.listdir(imgpath)
    run_counts = [int(d.split("_")[1]) for d in dirs if d.startswith("run")]
    run_counts.sort()
    run_count = run_counts[-1] if run_counts else 0

    imgpath += f'/run_{run_count+1}_{kwargs["prediction"]}_{kwargs["timesteps"]}{vae_inject_flag}'
    os.makedirs(imgpath)
        
    modelpath = f'./models/{kwargs["dataset"]}_{kwargs["degradation"]}{vae_flag}'
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    return imgpath, modelpath


def save_video(samples, save_dir, nrow, name="process.mp4"):
    """ Saves a video from Pytorch tensor 'samples'. 
    Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the video"""

    padding = 0
    imgs = []

    for idx in range(len(samples)):
        sample = samples[idx].cpu().detach().numpy()
        sample = np.clip(sample * 255, 0, 255)
        image_grid = make_grid(torch.Tensor(sample), nrow, padding=padding).numpy(
        ).transpose(1, 2, 0).astype(np.uint8)
        image_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)
        imgs.append(image_grid)

    video_size = tuple(reversed(tuple(s for s in imgs[0].shape[:2])))
    writer = cv2.VideoWriter(os.path.join(save_dir,name), cv2.VideoWriter_fourcc(*'mp4v'),
                             30, video_size)
    
    for i in range(len(imgs)):
        image = cv2.resize(imgs[i], video_size, fx=0,
                           fy=0, interpolation=cv2.INTER_CUBIC)
        writer.write(image)
    writer.release()


def save_gif(samples, save_dir, nrow, name="process.gif"):
    """ Saves a gif from Pytorch tensor 'samples'. Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the gif"""

    imgs = []

    for idx in range(len(samples)):
        s = samples[idx].cpu().detach().numpy()
        s = np.clip(s * 255, 0, 255).astype(np.uint8)
        image_grid = make_grid(torch.Tensor(s), nrow, padding=0)
        im = Image.fromarray(image_grid.permute(
            1, 2, 0).to('cpu', torch.uint8).numpy())
        imgs.append(im)

    imgs[0].save(os.path.join(save_dir,name), save_all=True,
                 append_images=imgs[1:], duration=0.5, loop=0)
    
    
def save_single_imgs(samples, save_dir, img_idx):

    idx = img_idx
    for img in samples:
        save_image(img, os.path.join(save_dir, f"{idx}.png"))
        idx += 1
    
    return idx
    

def load_dataset(batch_size = 32, dataset = 'mnist'):
    
    assert dataset in ['mnist', 'fashionmnist', 'cifar10', 'celeba', 'lsun_churches', 'afhq'],f"Invalid dataset, choose from ['mnist', 'fashionmnist', 'cifar10', 'celeba', 'lsun_churches', 'afhq']"

    # Check if directory exists
    if not os.path.exists(f'./data/{dataset.split("_")[0].upper()}'):
        os.makedirs(f'./data/{dataset.split("_")[0].upper()}')

    
    if dataset == 'mnist':

        training_data = datasets.MNIST(root='./data/MNIST', 
                                    train=True, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        val_data = datasets.MNIST(root='./data/MNIST', 
                                    train=False, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        
    elif dataset == 'fashionmnist':
            
        training_data = datasets.FashionMNIST(root='./data/FashionMNIST', 
                                    train=True, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        val_data = datasets.FashionMNIST(root='./data/FashionMNIST', 
                                    train=False, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
    
    elif dataset == 'cifar10':

        training_data = datasets.CIFAR10(root='./data/CIFAR10', 
                                    train=True, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        val_data = datasets.CIFAR10(root='./data/CIFAR10', 
                                    train=False, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
    
    elif dataset == 'celeba':
        
        train_transformation = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor()])
        
        scriptdir = os.path.dirname(__file__)
        datadir = os.path.join(scriptdir,'data')

        # Adapt path to data directory for DAS-6
        if 'scratch' in datadir:
            datadir = datadir.replace('MixedDiffusion/', '')

        print("Data Directory: ", datadir)

        training_data = MyCelebA(
            datadir,
            split='train',
            transform=train_transformation,
            download=False,
        )
        
        # Replace CelebA with your dataset
        val_data = MyCelebA(
            datadir,
            split='test',
            transform=train_transformation,
            download=False,
        )
    

    elif dataset == 'lsun_churches':
        try:
            scriptdir = os.path.dirname(__file__)
            datadir = os.path.join(scriptdir,'data/LSUN_CHURCHES')
            training_data = datasets.LSUN(root=datadir,
                                        classes=['church_outdoor_train'], 
                                        transform=T.Compose([
                                            T.Resize((256, 256)),
                                            T.ToTensor()]))
            val_data = datasets.LSUN(root=datadir,
                                        classes=['church_outdoor_val'], 
                                        transform=T.Compose([
                                            T.Resize((256, 256)),
                                            T.ToTensor()]))
        except:
            scriptdir = os.path.dirname(__file__)
            scriptdir = scriptdir.replace('MixedDiffusion/', 'experiments/')
            datadir = os.path.join(scriptdir,'data/LSUN_CHURCHES')
            training_data = datasets.LSUN(root=datadir,
                            classes=['church_outdoor_train'], 
                            transform=T.Compose([
                                T.Resize((256, 256)),
                                T.ToTensor()]))
            val_data = datasets.LSUN(root=datadir,
                                        classes=['church_outdoor_val'], 
                                        transform=T.Compose([
                                            T.Resize((256, 256)),
                                            T.ToTensor()]))
    
    elif dataset == 'afhq':
        train_loader = load_data(data_dir="./data/AFHQ_64/train",
                                batch_size=batch_size, image_size=64,
                                random_flip=False, num_workers=4)
        val_loader = load_data(data_dir="./data/AFHQ_64/test",
                               batch_size=batch_size, image_size=64,
                               random_flip=False, num_workers=4)
        
        return train_loader, val_loader


    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def plot_degradation(train_loader, **kwargs):

    timesteps = kwargs.pop('timesteps')
    kwargs.pop('degradation')
    noise = Degradation(timesteps = timesteps, degradation = 'noise', **kwargs)
    blur = Degradation(timesteps = timesteps, degradation = 'blur', **kwargs)
    black = Degradation(timesteps = timesteps, degradation = 'fadeblack', **kwargs)
    black_blur = Degradation(timesteps = timesteps, degradation = 'fadeblack_blur', **kwargs)
    blur_ban = Degradation(timesteps = timesteps, degradation = 'blur_bansal', **kwargs)
    black_blur_ban = Degradation(timesteps = timesteps, degradation = 'fadeblack_blur_bansal', **kwargs)

    #timesteps = min(50, timesteps)

    plt.figure(figsize=(16, 10))
    for i, j in enumerate(range(1, timesteps + 1, 10)):

        ind = i + 1
        x, y = next(iter(train_loader)) 
        t = torch.tensor([j]).repeat(x.shape[0],).to('mps')
        x = x.to('mps')

        plt.subplot(7, timesteps//10, 0*timesteps//10+ind)
        x_plain = x[0].unsqueeze(0) if len(x[0].shape) == 2 else x
        plt.imshow(x_plain[0].cpu().permute(1, 2, 0))
        plt.axis('off')

        plt.subplot(7, timesteps//10, 1*timesteps//10+ind)
        x_noise = noise.degrade(x, t).cpu()
        x_noise = x_noise[0].unsqueeze(0) if len(x_noise[0].shape) == 2 else x_noise[0]
        plt.imshow(x_noise.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
    
        plt.subplot(7, timesteps//10, 2*timesteps//10+ind)
        x_blur = blur.degrade(x, t).cpu()
        x_blur = x_blur[0].unsqueeze(0) if len(x_blur[0].shape) == 2 else x_blur[0]
        plt.imshow(x_blur.permute(1, 2, 0), vmin=0, vmax=1)
        plt.axis('off')
        
        plt.subplot(7, timesteps//10, 3*timesteps//10+ind)
        x_black = black.degrade(x, t).cpu()
        x_black = x_black[0].unsqueeze(0) if len(x_black[0].shape) == 2 else x_black[0]
        plt.imshow(x_black.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
    
        plt.subplot(7, timesteps//10, 4*timesteps//10+ind)
        x_blackblur = black_blur.degrade(x, t).cpu()
        x_blackblur = x_blackblur[0].unsqueeze(0) if len(x_blackblur[0].shape) == 2 else x_blackblur[0]
        plt.imshow(x_blackblur.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')

        plt.subplot(7, timesteps//10, 5*timesteps//10+ind)
        x_blur_ban = blur_ban.degrade(x, t).cpu()
        x_blur_ban = x_blur_ban[0].unsqueeze(0) if len(x_blur_ban[0].shape) == 2 else x_blur_ban[0]
        plt.imshow(x_blur_ban.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')

        plt.subplot(7, timesteps//10, 6*timesteps//10+ind)
        x_blackblur_ban = black_blur_ban.degrade(x, t).cpu()
        x_blackblur_ban = x_blackblur_ban[0].unsqueeze(0) if len(x_blackblur_ban[0].shape) == 2 else x_blackblur_ban[0]
        plt.imshow(x_blackblur_ban.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
        

    # axis off
    plt.suptitle('Image degradation', size = 18)


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True
    


