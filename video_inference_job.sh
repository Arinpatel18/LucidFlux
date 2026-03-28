#!/bin/bash
#SBATCH --job-name=lucidflux_video_infer
#SBATCH --output=/scratch/data/avinash1/LucidFlux/video_infer_%j.out
#SBATCH --error=/scratch/data/avinash1/LucidFlux/video_infer_%j.err

#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G

#SBATCH --time=04:00:00

echo "===== VIDEO INFERENCE JOB START ====="
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

echo "Running video inference..."
# Edit the --input_folder to point to your frames directory
python video_inference.py \
  --checkpoint weights/lucidflux/lucidflux.pth \
  --input_folder "assests/input.mp4" \
  --output_folder "video_results" \
  --width 512 \
  --height 512 \
  --num_steps 20 \
  --swinir_pretrained weights/swinir.pth \
  --siglip_ckpt weights/siglip \
  --offload

echo "===== VIDEO INFERENCE DONE ====="