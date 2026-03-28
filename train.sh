#!/usr/bin/env bash

# source weights/env.sh
export FLUX_DEV_FLOW=/scratch/data/avinash1/LucidFlux/weights/flux-dev/flux1-dev.safetensors
export FLUX_DEV_AE=/scratch/data/avinash1/LucidFlux/weights/flux-dev/ae.safetensors
export HF_HOME=/scratch/data/avinash1/huggingface1₹

log_dir="train_logs"
timestamp="$(date +%Y%m%d%H%M)"
log_file="${log_dir}/train_log_${timestamp}.log"

mkdir -p "${log_dir}"

accelerate launch --config_file "train_configs/ds_zero2.yaml" train.py \
                --config "train_configs/train_LucidFlux.yaml" \
                2>&1 | tee "${log_file}"