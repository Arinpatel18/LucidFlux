#!/usr/bin/env bash
set -e

# Env variables are set in inference_job.sh
# source weights/env.sh

python inference.py \
  --checkpoint Lucidflux_final.pth \
  --control_image assets/1.png \
  --output_dir /scratch/data/avinash1/LucidFlux/outputs-trained \
  --width 512 \
  --height 512 \
  --num_steps 20 \
  --swinir_pretrained weights/swinir.pth \
  --siglip_ckpt weights/siglip \
  --offload
