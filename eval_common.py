"""
Shared helpers for evaluate.py.
"""

import os
import glob
import random
import math

import torch
import numpy as np

import utils3d
from skimage.metrics import structural_similarity as skimage_ssim


def find_ckpt(cfg):
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_random_camera(seed, device='cuda'):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    yaw = torch.rand(1, generator=gen).item() * 2 * math.pi
    pitch = (torch.rand(1, generator=gen).item() - 0.5) * (math.pi / 3) + math.pi / 12 # roughly -15 to +45 deg
    radius = torch.rand(1, generator=gen).item() * 1.5 + 2.5 # 2.5 to 4.0
    fov = 40.0

    cx = math.sin(yaw) * math.cos(pitch)
    cy = math.cos(yaw) * math.cos(pitch)
    cz = math.sin(pitch)
    origin = torch.tensor([[cx, cy, cz]], dtype=torch.float32, device=device) * radius

    extrinsics = utils3d.torch.extrinsics_look_at(origin, torch.tensor([[0.0, 0.0, 0.0]], device=device), torch.tensor([[0.0, 0.0, 1.0]], device=device))
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor([math.radians(fov)], device=device), torch.tensor([math.radians(fov)], device=device))
    return extrinsics[0], intrinsics[0]


def compute_ssim(img1, img2):
    """img1, img2: torch tensors [C, H, W] in [0, 1]"""
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return float(skimage_ssim(img1_np, img2_np, channel_axis=2, data_range=1.0))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
