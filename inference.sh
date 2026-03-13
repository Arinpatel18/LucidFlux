#!/usr/bin/env bash
set -e

# Minimal, readable, zero-logic wrapper. All paths are relative.
# Ensure you've prepared weights beforehand (e.g., source weights/env.sh for FLUX base).

source weights/env.sh

python inference.py \
  --checkpoint weights/lucidflux/lucidflux.pth \
  --control_image assets/3.png \
  --output_dir outputs \
  --width 1024 \
  --height 1024 \
  --num_steps 20 \
  --swinir_pretrained weights/swinir.pth \
  --siglip_ckpt weights/siglip \
  --offload

python inference-2k.py \
  --checkpoint weights/lucidflux/lucidflux.pth \
  --control_image assets/3.png \
  --output_dir outputs-2k \
  --width 2048 \
  --height 2048 \
  --num_steps 20 \
  --swinir_pretrained weights/swinir.pth \
  --siglip_ckpt weights/siglip \
  --offload
