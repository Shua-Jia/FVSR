
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils import data
import math

# -------------------------- 复用训练文件中的模型定义 --------------------------
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

        dim_t = torch.arange(C // 2, device=device).float()
        omega = 1.0 / (10000 ** (2 * dim_t / C))

        sin_h = torch.sin(h_idx[:, None] * omega[None, :])
        cos_h = torch.cos(h_idx[:, None] * omega[None, :])
        pos_h = torch.cat([sin_h, cos_h], dim=1)

        sin_w = torch.sin(w_idx[:, None] * omega[None, :])
        cos_w = torch.cos(w_idx[:, None] * omega[None, :])
        pos_w = torch.cat([sin_w, cos_w], dim=1)

        pos_h = pos_h.permute(1, 0)[:, :, None]
        pos_w = pos_w.permute(1, 0)[:, None, :]
        pos_enc = pos_h + pos_w

        return pos_enc[None]


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
        self.pos_encoder = SinusoidalPositionEncoding2D(out_channels)

        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bnorm1(self.relu(x))
        residual = x
        x = self.norm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        pos_enc = self.pos_encoder(q)
        q = q + pos_enc
        k = k + pos_enc

        B, C, H, W = q.shape
        N = H * W

        q = q.view(B, self.num_heads, self.head_dim, N)
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)

        attn_weights = torch.matmul(q.transpose(-1, -2), k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v.transpose(-1, -2))
        out = out.transpose(-1, -2).contiguous()

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
        self.attn_up = AttentionBlock(c, c, num_heads)
        self.attn_down = AttentionBlock(c, c, num_heads)
        self.res_scale = nn.Parameter(torch.ones(loops))

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = x + time_emb[:, :, None, None]
        out = x
        for i in range(self.loops):
            residual = out
            out = self.attn_up(out)
            out = self.attn_down(out)
            out = out + residual * self.res_scale[i]
        return out


# -------------------------- DDIM采样核心逻辑（适配Latent空间） --------------------------
class LatentDDIMSampler:
    def __init__(self, noise_predictor, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"):
        self.noise_predictor = noise_predictor
        self.num_timesteps = num_timesteps
        self.device = device

        # 初始化扩散参数（与训练一致）
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def ddim_sample_step(self, x, t, t_prev, latent_img_lr, eta=0.0):
        """单步DDIM采样（适配Latent空间，融合low-res latent）"""
        B = x.shape[0]
        if isinstance(t, int):
            t = torch.full((B,), t, device=self.device, dtype=torch.long)
        if isinstance(t_prev, int):
            t_prev = torch.full((B,), t_prev, device=self.device, dtype=torch.long)

        # 融合当前x（带噪声latent）和low-res latent（与训练时一致：x_t + latent_img_lr）
        combined_latent = x + latent_img_lr

        # 预测噪声
        eps_pred = self.noise_predictor(combined_latent, t)

        # 获取扩散参数
        alpha_t = self.extract(self.alphas_cumprod, t, x.shape)
        if t_prev[0] >= 0:
            alpha_t_prev = self.extract(self.alphas_cumprod, t_prev, x.shape)
        else:
            alpha_t_prev = torch.ones_like(alpha_t)

        # DDIM方差计算
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

        # 预测x0
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)

        # 计算下一步mean和sample
        if t_prev[0] >= 0:
            mean = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * eps_pred
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = pred_x0  # 最后一步直接输出x0

        return x_prev

    @torch.no_grad()
    def ddim_sample_loop(self, latent_img_lr, num_ddim_steps=100, eta=0.0):
        """
        完整DDIM采样流程
        :param latent_img_lr: 低分辨率/模糊图片的latent (B, C, H, W)
        :param num_ddim_steps: DDIM采样步数
        :param eta: 0=确定性DDIM，1=类DDPM随机
        :return: 去噪后的latent
        """
        B, C, H, W = latent_img_lr.shape
        # 从纯噪声开始
        x = torch.randn_like(latent_img_lr, device=self.device)

        # 生成DDIM采样的时间步（均匀采样）
        skip = self.num_timesteps // num_ddim_steps
        timesteps = list(range(0, self.num_timesteps, skip))
        if timesteps[-1] != self.num_timesteps - 1:
            timesteps.append(self.num_timesteps - 1)
        timesteps = sorted(timesteps, reverse=True)

        # 逐步采样
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.ddim_sample_step(x, t, t_prev, latent_img_lr, eta=eta)

        return x


def downgrade_inference(x, org_size, scale_size, target_domain, domains_interval):
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
# 128
#E:\AI_Lab\lab25 FVSR General Eval\input\frames128 文件夹下包含50个文件夹，每个文件夹中包含多个图像，图像的大小都是128，
# 对其中的任意9个文件夹添加iaa.GaussianBlur((4)), 对其中的任意9个文件夹添加iaa.AverageBlur((7)),对其中的任意8个文件夹添加iaa.MedianBlur((9)),
# 对其中的任意8个文件夹添加iaa.MotionBlur((5)),,对其中的任意8个文件夹添加iaa.MeanShiftBlur((6)),

# 256
# E:\AI_Lab\lab25 FVSR General Eval\input\frames256 文件夹下包含50个文件夹，
# 每个文件夹中包含多个图像，图像的大小都是128， 对其中的任意5个文件夹添加iaa.GaussianBlur((4)),
# 对其中的任意5个文件夹添加iaa.AverageBlur((7)),对其中的任意5个文件夹添加iaa.MedianBlur((9)),
# 对其中的任意5个文件夹添加iaa.MotionBlur((5)),,对其中的任意5个文件夹添加iaa.MeanShiftBlur((6))，
# 对其中的任意5个文件夹添加iaa.MotionBlur((5))与iaa.MeanShiftBlur((6))，对其中的任意5个文件夹添加iaa.MotionBlur((5))与iaa.MedianBlur((9)),
# 对其中的任意5个文件夹添加iaa.AverageBlur((7))与iaa.MedianBlur((9)), 对其中的任意5个文件夹添加iaa.AverageBlur((7))与iaa.GaussianBlur((4)),
# 任意5个与任意5个之间不重复，并按照类别，输出加不同模糊后的文件夹名称，如GaussianBlur： Ip4QAn2yg0A_377_0，mxWGCR-Xg3M_14_0



# -------------------------- 数据处理函数 --------------------------
def downgrade_inference(x, target_size, domain_list, domains_interval):
    """推理时的图片降级（与训练时downgrade_training逻辑对齐，固定参数）"""
    org_size = domain_list[0]
    #scale_size = domain_list[target_size]

    x = np.array(x.add(1).mul(255 / 2))
    x = np.uint8(x)

    # 推理时固定模糊参数（可根据需求调整）
    # if target_size == domains_interval.index(768):
    #     num_blurs = 2
    # elif target_size == domains_interval.index(512):
    #     num_blurs = 2
    # elif target_size == domains_interval.index(384):
    #     num_blurs = 1
    # elif target_size == domains_interval.index(256):
    #     num_blurs = 1
    # elif target_size == domains_interval.index(128):
    #     num_blurs = 0
    # elif target_size == domains_interval.index(64):
    #     num_blurs = 0
    # else:
    #     num_blurs = 0
    #
    # all_blurs = [
    #     iaa.GaussianBlur((4, 6)),
    #     iaa.AverageBlur((7, 9)),
    #     iaa.MedianBlur((9, 11)),
    #     iaa.MotionBlur((5, 7)),
    #     iaa.MeanShiftBlur(spatial_radius=(6.0, 7.0), color_radius=(6.0, 7.0)),
    # ]
    # selected_blurs = all_blurs[:num_blurs] if num_blurs > 0 else []
    #
    # aug_seq = iaa.Sequential([
    #     iaa.Sometimes(1.0, iaa.OneOf(selected_blurs)) if num_blurs > 0 else iaa.Identity(),
    #     iaa.Resize(org_size, interpolation=ia.ALL),
    #     iaa.JpegCompression(compression=(30, 60)),
    #     iaa.Resize(org_size, interpolation=ia.ALL),
    # ])
    # aug_img = aug_seq(images=x)
    return torch.from_numpy(x).mul(2 / 255).add(-1)


class InferenceDataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        img_name = path.name
        return self.transform(img), img_name