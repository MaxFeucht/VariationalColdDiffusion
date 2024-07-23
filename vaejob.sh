#!/bin/bash
#SBATCH --job-name=vcd_pix4
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH -C A4000
#SBATCH --ntasks-per-node=1 
#SBATCH --partition=defq
#SBATCH --gres=gpu:1 
#SBATCH -o vcd_pix4.out   

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
batch_size=32
timesteps=8
dim=64
epochs=1000
prediction="vxt"
degradation="black_pixelation"
noise_schedule="cosine" 
dataset="afhq" 
sample_interval=5
n_samples=32
model_ema_decay=0.998
vae_beta=0.004
noise_scale=0.005
latent_dim=512
vae_inject="concat"
vae_loc="emb"
xt_dropout=0
min_t2_step=1
kl_annealing=10000
baseline="xx"
 
# Run the actual experiment. 
python /var/scratch/mft520/MixedDiffusion/main.py --epochs $epochs --batch_size $batch_size --timesteps $timesteps --dim $dim \
                                                --lr $lr --prediction $prediction --degradation $degradation \
                                                --noise_schedule $noise_schedule --dataset $dataset --sample_interval $sample_interval \
                                                --n_samples $n_samples --model_ema_decay $model_ema_decay --noise_scale $noise_scale \
                                                --latent_dim $latent_dim --vae_beta $vae_beta --vae_inject $vae_inject --vae_loc $vae_loc \
                                                --min_t2_step $min_t2_step --var_sampling_step $min_t2_step --baseline $baseline \
                                                --xt_dropout $xt_dropout --cluster --xt_dropout $xt_dropout --kl_annealing $kl_annealing \
                                                --cluster --vae --cold_perturb

echo "Script finished"
