
import numpy as np
import argparse
import imgaug.augmenters as iaa
import imgaug as ia
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

from torch.utils import data

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

domains_interval = [1024, 768, 512, 384, 256, 128, 64]


class SinusoidalPositionEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        assert channels % 2 == 0, "Channels must be even for sinusoidal encoding"

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels

        device = x.device
        h_idx = torch.arange(H, device=device).float()
        w_idx = torch.arange(W, device=device).float()

        # Frequency for all C dimensions
        dim_t = torch.arange(C // 2, device=device).float()
        omega = 1.0 / (10000 ** (2 * dim_t / C))  # [C//2]

        # H encoding: [H, C//2] -> expand to [H, C]
        sin_h = torch.sin(h_idx[:, None] * omega[None, :])  # [H, C//2]
        cos_h = torch.cos(h_idx[:, None] * omega[None, :])  # [H, C//2]
        pos_h = torch.cat([sin_h, cos_h], dim=1)  # [H, C]

        # W encoding: [W, C//2] -> [W, C]
        sin_w = torch.sin(w_idx[:, None] * omega[None, :])
        cos_w = torch.cos(w_idx[:, None] * omega[None, :])
        pos_w = torch.cat([sin_w, cos_w], dim=1)  # [W, C]

        # Reshape and add
        pos_h = pos_h.permute(1, 0)[:, :, None]   # [C, H, 1]
        pos_w = pos_w.permute(1, 0)[:, None, :]   # [C, 1, W]
        pos_enc = pos_h + pos_w  # [C, H, W]

        return pos_enc[None]  # [1, C, H, W]


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.q_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)

        # 固定正弦位置编码（无参数）
        self.pos_encoder = SinusoidalPositionEncoding2D(out_channels)

        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bnorm1(self.relu(x))
        residual = x
        x = self.norm(x)

        # 注：位置编码应加在 Q/K/V 投影前 or 后？通常加在输入上。
        # 这里我们加在 norm 之后、投影之前（也可加在 x 上再 norm，差别不大）
        # 但注意：pos encoding 应与后续投影的通道一致 → 所以加在 out_channels 维度
        # 然而 x 是 in_channels，所以我们不能直接加！

        # ✅ 正确做法：先做 Q/K/V 投影，再加位置编码（因为 PE 维度 = out_channels）
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 获取位置编码（基于 q 的空间尺寸）
        pos_enc = self.pos_encoder(q)  # [1, out_channels, H, W]

        # 将位置编码加到 Q 和 K（常见做法：只加到 Q 和 K，不加 V）
        q = q + pos_enc
        k = k + pos_enc
        # v 保持不变

        B, C, H, W = q.shape
        N = H * W

        # Reshape for multi-head attention: [B, num_heads, head_dim, N]
        q = q.view(B, self.num_heads, self.head_dim, N)
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q.transpose(-1, -2), k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v.transpose(-1, -2))  # [B, num_heads, N, head_dim]
        out = out.transpose(-1, -2).contiguous()  # [B, num_heads, head_dim, N]

        out = out.view(B, self.num_heads * self.head_dim, H, W)
        out = self.out_proj(out)
        out = self.bnorm2(self.relu(out))

        return out + residual

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class NoisePredictor(nn.Module):
    def __init__(self, c, num_heads=4, loops=3, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, c),
            nn.SiLU(),
            nn.Linear(c, c)
        )
        self.loops = loops
        self.attn_up = AttentionBlock(c, c , num_heads)
        self.attn_down = AttentionBlock(c , c, num_heads)
        self.res_scale = nn.Parameter(torch.ones(loops))

    def forward(self, x, t):
        time_emb = self.time_mlp(t)  # (B, C)
        # Broadcast to (B, C, 1, 1) and add
        x = x + time_emb[:, :, None, None]
        out = x
        for i in range(self.loops):
            residual = out
            out = self.attn_up(out)
            out = self.attn_down(out)
            out = out + residual * self.res_scale[i]
        return out



def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

transform_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
trans = transforms.Compose(transform_list)


def downgrade_training(x, target_size, domain_list):
    org_size = domain_list[0]
    for j in range(x.shape[0]):
        scale_size = domain_list[target_size[j]]
        x[j] = random_augment(x[j], org_size, scale_size,target_size[j], domains_interval)
    return x

def random_augment(x, org_size, scale_size, target_domain, domains_interval):
    """Input single RGB PIL Image instance"""
    x = np.array(x.add(1).mul(255 / 2))
    x = np.uint8(x)

    if target_domain == domains_interval.index(768):
        blur_probs = [0.5, 0.4, 0.08, 0.02]  # [5种, 4种, 3种, 2种] [0.4, 0.3, 0.2, 0.1]
        blur_choices = [5, 4, 3, 2]
    elif target_domain == domains_interval.index(512):
        blur_probs = [0.5, 0.4, 0.08, 0.02]  # [4种, 3种, 2种, 1种] [0.4, 0.3, 0.2, 0.1]
        blur_choices = [4, 3, 2, 1]
    elif target_domain == domains_interval.index(384):
        blur_probs = [0.5, 0.45, 0.05]  # [3种, 2种, 1种] [0.4, 0.4, 0.2]
        blur_choices = [3, 2, 1]
    elif target_domain == domains_interval.index(256):
        blur_probs = [0.9, 0.1]  # [2种, 1种]
        blur_choices = [2, 1]
    elif target_domain == domains_interval.index(128):
        blur_probs = [1.0]  # [1种]
        blur_choices = [1]
    elif target_domain == domains_interval.index(64):
        blur_probs = [1.0]  # [0种]
        blur_choices = [0]
    else:
        blur_probs = [1.0]  # [0种]
        blur_choices = [0]

    num_blurs = np.random.choice(blur_choices, p=blur_probs)

    all_blurs = [
        iaa.GaussianBlur((4, 6)), # (3,7)
        iaa.AverageBlur((7, 9)),  #(6,10)
        iaa.MedianBlur((9,11)),   # (7,11)
        iaa.MotionBlur((5, 7)),  # or (4,8)
        iaa.MeanShiftBlur(spatial_radius=(6.0,7.0), color_radius=(6.0,7.0)),  # spatial_radius=10.0, color_radius=10.0
    ]

    selected_blurs = np.random.choice(all_blurs, size=num_blurs, replace=False)

    aug_seq = iaa.Sequential([
        iaa.Sometimes(0.999, iaa.OneOf(selected_blurs)),
        iaa.Resize(scale_size, interpolation=ia.ALL),
        iaa.Sometimes(0.9, iaa.JpegCompression(compression=(10, 65))),
        iaa.Resize(org_size, interpolation=ia.ALL),
    ])
    aug_img = aug_seq(images=x)
    return torch.from_numpy(aug_img).mul(2 / 255).add(-1)


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)