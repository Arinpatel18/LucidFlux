import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def _scan_images(root_dir: Path):
    files = {}
    for path in sorted(root_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(root_dir)
        key = rel.with_suffix('').as_posix()
        files[key] = path
    return files


class LQGTPairedDataset(Dataset):
    def __init__(self, data_root, img_size=512):
        if data_root is None:
            raise ValueError('必须提供 data_root，且其下包含 lq/ 和 gt/ 子目录')

        self.data_root = Path(data_root)
        self.lq_root = self.data_root / 'lq'
        self.gt_root = self.data_root / 'gt'
        if not self.lq_root.is_dir() or not self.gt_root.is_dir():
            raise FileNotFoundError(f'{self.data_root} 下必须同时存在 lq/ 和 gt/ 子目录')

        lq_files = _scan_images(self.lq_root)
        gt_files = _scan_images(self.gt_root)
        keys = sorted(set(lq_files) & set(gt_files))
        if not keys:
            raise FileNotFoundError(f'在 {self.data_root} 下没有找到可配对的 lq/gt 图像')

        self.pairs = [(gt_files[key], lq_files[key]) for key in keys]
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, lq_path = self.pairs[idx]
        gt = Image.open(gt_path).convert('RGB')
        lq = Image.open(lq_path).convert('RGB')

        gt = c_crop(gt).resize((self.img_size, self.img_size))
        lq = c_crop(lq).resize((self.img_size, self.img_size))

        gt = torch.from_numpy((np.array(gt) / 127.5) - 1).permute(2, 0, 1)
        hint = torch.from_numpy((np.array(lq) / 127.5) - 1).permute(2, 0, 1)
        return gt, hint


def loader(train_batch_size, num_workers, **args):
    dataset = LQGTPairedDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
