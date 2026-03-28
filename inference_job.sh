#!/bin/bash
#SBATCH --job-name=lucidflux_infer
#SBATCH --output=/scratch/data/avinash1/LucidFlux/infer_%j.out
#SBATCH --error=/scratch/data/avinash1/LucidFlux/infer_%j.err

#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G

#SBATCH --time=04:00:00

echo "===== INFERENCE JOB START ====="
hostname
date

# Load conda
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate lucidflux

# Env variables
export FLUX_DEV_FLOW=/scratch/data/avinash1/LucidFlux/weights/flux-dev/flux1-dev.safetensors
export FLUX_DEV_AE=/scratch/data/avinash1/LucidFlux/weights/flux-dev/ae.safetensors
export HF_HOME=/scratch/data/avinash1/huggingface

cd /iitjhome/avinash1/LucidFlux

echo "GPU info:"
nvidia-smi

echo "Running inference..."
bash inference.sh

echo "===== INFERENCE DONE ====="
