#!/bin/bash
#SBATCH --job-name=sampler_afhq
#SBATCH --time=14:00:00 
#SBATCH -N 1
#SBATCH -C A5000
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH -o sampler_afhq.out

## in the list above, the partition name depends on where you are running your job. 
## On DAS5 the default would be `defq` on Lisa the default would be `gpu` or `gpu_shared`
## Typing `sinfo` on the server command line gives a column called PARTITION.  There, one can find the name of a specific node, the state (down, alloc, idle etc), the availability and how long is the time limit . Ask your supervisor before running jobs on queues you do not know.

# Load GPU drivers 

## Enable the following two lines for DAS5
# module load cuda10.0/toolkit
# module load cuDNN/cuda10.0

## Enable the following line for DAS6
module load cuda11.3/toolkit/11.3.1

## For Lisa and Snellius, modules are usually not needed
## https://userinfo.surfsara.nl/systems/shared/modules 

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate

# Scratch directory has far more space than home directory.
mkdir /var/scratch/mft520/experiments
cd /var/scratch/mft520/experiments

# # Base directory for the experiment
# mkdir $HOME/experiments
# cd $HOME/experiments

# Simple trick to create a unique directory for each run of the script
# echo $$
# mkdir o`echo $$`
# cd o`echo $$`

## Set Vars

lr=1e-4
batch_size=128
timesteps=200
dim=128
epochs=1000
prediction="vxt"
degradation="fadeblack_blur"
noise_schedule="cosine" 
dataset="afhq" 
sample_interval=1
n_samples=60
model_ema_decay=0.997
vae_alpha=0.9999
noise_scale=0.01
latent_dim=32
vae_inject="add"
vae_loc="start"
xt_dropout=0.1
min_t2_step=25
sampling_steps=(25 50 75 100 125 150 175 200)


# Run the actual experiment. 
python /var/scratch/mft520/MixedDiffusion/sampler.py --epochs $epochs --batch_size $batch_size --timesteps $timesteps --dim $dim \
                                                --lr $lr --prediction $prediction --degradation $degradation \
                                                --noise_schedule $noise_schedule --dataset $dataset --sample_interval $sample_interval \
                                                --n_samples $n_samples --model_ema_decay $model_ema_decay --noise_scale $noise_scale \
                                                --latent_dim $latent_dim --vae_alpha $vae_alpha --vae_inject $vae_inject --vae_loc $vae_loc \
                                                --min_t2_step $min_t2_step --sampling_steps $sampling_steps \
                                                --xt_dropout $xt_dropout --cluster --vae --loss_weighting
echo "Script finished"
