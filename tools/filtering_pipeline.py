#!/usr/bin/env python3
"""
Usage:
python single_image_scores.py /path/to/image.jpg
python single_image_scores.py /path/to/image.jpg --clipiqa-device cpu
python single_image_scores.py /path/to/image.jpg --output /path/to/result.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch
from PIL import Image


FLAT_PATCH_SIZE = 240
FLAT_VARIANCE_THRESHOLD = 800.0
_CLIPIQA_METRICS = {}


def safe_imread(image_path: str) -> np.ndarray:
    with Image.open(image_path) as pil_img:
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        image = np.array(pil_img)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def calculate_blur_score(image: np.ndarray) -> float:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def get_flat_percentage(image: np.ndarray) -> float:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape
    flat_patches = 0
    total_patches = 0

    for y in range(0, h, FLAT_PATCH_SIZE):
        for x in range(0, w, FLAT_PATCH_SIZE):
            patch = gray[y:y + FLAT_PATCH_SIZE, x:x + FLAT_PATCH_SIZE]
            if patch.shape[0] != FLAT_PATCH_SIZE or patch.shape[1] != FLAT_PATCH_SIZE:
                continue

            total_patches += 1
            sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

            if magnitude.var() < FLAT_VARIANCE_THRESHOLD:
                flat_patches += 1

    if total_patches == 0:
        return 0.0
    return float(flat_patches / total_patches)


def get_clipiqa_metric(device: str):
    metric = _CLIPIQA_METRICS.get(device)
    if metric is None:
        metric = pyiqa.create_metric('clipiqa', device=device)
        _CLIPIQA_METRICS[device] = metric
    return metric


def get_clipiqa_score(image_path: str, device: str) -> float:
    metric = get_clipiqa_metric(device)
    score = metric(image_path)
    if isinstance(score, torch.Tensor):
        score = score.item()
    return float(score)


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute blur, flat, and clipiqa scores for a single image.')
    ap.add_argument('image_path', help='Path to the image file')
    ap.add_argument('--clipiqa-device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output', default='', help='Optional output json path')
    args = ap.parse_args()

    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f'image not found: {image_path}')

    image = safe_imread(str(image_path))
    result = {
        'blur_score': calculate_blur_score(image),
        'flat_percentage': get_flat_percentage(image),
        'clipiqa_score': get_clipiqa_score(str(image_path), args.clipiqa_device),
    }

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
