#!/bin/bash
#SBATCH --job-name=lucidflux_train
#SBATCH --output=/scratch/data/avinash1/LucidFlux/train_%j.out
#SBATCH --error=/scratch/data/avinash1/LucidFlux/train_%j.err

#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G

#SBATCH --time=24:00:00

echo "========== JOB START =========="
echo "Hostname:"
hostname
echo "Date:"
date

# 🔥 Load conda properly
echo "Loading conda..."
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh

echo "Activating env..."
conda activate lucidflux

echo "Python path:"
which python

echo "Checking GPU:"
nvidia-smi

# 🔥 Set env vars
export FLUX_DEV_FLOW=/scratch/data/avinash1/LucidFlux/weights/flux-dev/flux1-dev.safetensors
export FLUX_DEV_AE=/scratch/data/avinash1/LucidFlux/weights/flux-dev/ae.safetensors
export HF_HOME=/scratch/data/avinash1/huggingface

cd /iitjhome/avinash1/LucidFlux

echo "Starting training..."
bash train.sh

echo "========== JOB END =========="