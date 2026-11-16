import math
from typing import Optional, Tuple, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================
# DDIM Scheduler + Sampler
# ===============================
class DDIMScheduler:
    """
    A minimal DDIM scheduler that works with tensors shaped [B, C, H, W] of arbitrary C/H/W.

    prediction_type:
        - "epsilon": model predicts noise eps.
        - "v_prediction": model predicts v as in Imagen/StableDiffusion-v2.
    """
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",     # "linear" or "cosine"
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.prediction_type = prediction_type
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=self.dtype)
        elif beta_schedule == "cosine":
            betas = self._betas_for_alpha_bar(self.num_train_timesteps, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = None  # will be set by set_timesteps

    @staticmethod
    def _betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999, dtype=torch.float32):
        # cosine schedule from Nichol & Dhariwal
        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            beta = min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
            betas.append(beta)
        return torch.tensor(betas, dtype=dtype)

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None):
        device = torch.device(device)
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        for name in ["betas", "alphas", "alphas_cumprod"]:
            tensor = getattr(self, name).to(device=device, dtype=self.dtype)
            setattr(self, name, tensor)
        if self.timesteps is not None:
            self.timesteps = self.timesteps.to(device=device)
        return self

    @torch.no_grad()
    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """
        Create a descending list of timesteps of length num_inference_steps from the training grid.
        """
        device = device or self.device
        t = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        self.timesteps = t

    def _get_alpha(self, t: int) -> torch.Tensor:
        return self.alphas_cumprod[t]

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """
        Extract a[t] and reshape to [B, 1, 1, 1] for broadcasting.
        a: [T], t: [B]
        """
        out = a.gather(0, t.clamp(min=0, max=a.shape[0]-1))
        return out.reshape(-1, 1, 1, 1)

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,  # predicted noise eps or v
        t_index: int,                # index in the self.timesteps list
        sample: torch.Tensor,        # x_t
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one DDIM step: x_t -> x_{t-1} (or x_{t_prev} based on schedule).
        Returns (prev_sample, pred_x0).
        """
        assert self.timesteps is not None, "Call set_timesteps(...) first."
        b, c, h, w = sample.shape
        device = sample.device

        # current and prev timesteps (on the training grid)
        t_cur = int(self.timesteps[t_index].item())
        if t_index == len(self.timesteps) - 1:
            alpha_prev = torch.tensor(1.0, device=device, dtype=self.dtype)
        else:
            t_prev = int(self.timesteps[t_index + 1].item())
            alpha_prev = self._get_alpha(t_prev).to(device=device, dtype=self.dtype)

        alpha_t = self._get_alpha(t_cur).to(device=device, dtype=self.dtype)

        # Make [B,1,1,1] versions
        alpha_t_b = alpha_t.view(1, 1, 1, 1).expand(b, 1, 1, 1)
        alpha_prev_b = alpha_prev.view(1, 1, 1, 1).expand(b, 1, 1, 1)

        # Compute eps and x0 depending on prediction_type
        if self.prediction_type == "epsilon":
            eps = model_output
            pred_x0 = (sample - torch.sqrt(1 - alpha_t_b) * eps) / torch.sqrt(alpha_t_b + 1e-12)
        elif self.prediction_type == "v_prediction":
            v = model_output
            # x0 = sqrt(a)*x - sqrt(1-a)*v; eps = (x - sqrt(a)*x0)/sqrt(1-a)
            pred_x0 = torch.sqrt(alpha_t_b) * sample - torch.sqrt(1 - alpha_t_b) * v
            eps = (sample - torch.sqrt(alpha_t_b) * pred_x0) / torch.sqrt(1 - alpha_t_b + 1e-12)
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        # Compute DDIM sigma and direction term
        denom = torch.clamp(1 - alpha_t_b, min=1e-12)
        ratio = torch.clamp(alpha_t_b / torch.clamp(alpha_prev_b, min=1e-12), min=0.0, max=1.0)
        sigma = eta * torch.sqrt(torch.clamp((1 - alpha_prev_b) / denom, min=0.0)) * torch.sqrt(torch.clamp(1 - ratio, min=0.0))
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_prev_b - sigma**2, min=0.0)) * eps

        # random noise
        if eta > 0.0:
            noise = torch.randn(sample.shape, dtype=sample.dtype, device=sample.device, generator=generator)
        else:
            noise = torch.zeros_like(sample)

        x_prev = torch.sqrt(alpha_prev_b) * pred_x0 + dir_xt + sigma * noise
        return x_prev, pred_x0


@torch.no_grad()
def ddim_sample(
    model: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    shape: Tuple[int, int, int, int],
    num_inference_steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    cond: Optional[torch.Tensor] = None,
    uncond_cond: Optional[torch.Tensor] = None,
    num_train_timesteps: int = 1000,
    beta_schedule: str = "linear",
    prediction_type: str = "epsilon",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    x_T: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run DDIM sampling.

    model: Callable(x_t, t, cond) -> predicted noise (or v if prediction_type='v_prediction').
    shape: (B, C, H, W)
    Returns: (final_sample, last_pred_x0)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    B, C, H, W = shape
    if generator is None:
        generator = torch.Generator(device=device)

    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        device=device,
        dtype=dtype,
    ).to(device, dtype=dtype)
    scheduler.set_timesteps(num_inference_steps, device=device)

    if x_T is None:
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    else:
        x = x_T.to(device=device, dtype=dtype)
        assert x.shape == shape, f"x_T shape {tuple(x.shape)} != expected shape {shape}"

    last_pred_x0 = None

    for i, t_scalar in enumerate(scheduler.timesteps):
        t_batch = torch.full((B,), int(t_scalar.item()), device=device, dtype=torch.long)

        # Classifier-free guidance (optional)
        if guidance_scale is not None and guidance_scale != 1.0 and uncond_cond is not None:
            eps_uncond = model(x, t_batch, uncond_cond)
            eps_cond = model(x, t_batch, cond)
            model_out = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            model_out = model(x, t_batch, cond)

        x, last_pred_x0 = scheduler.step(model_out, i, x, eta=eta, generator=generator)

    return x, last_pred_x0


# ===============================
# Tiny DiT-style Transformer model
# ===============================
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """
    Sinusoidal timestep embedding. timesteps: [B] (long or float), returns [B, dim]
    """
    if timesteps.dtype in (torch.int32, torch.int64):
        timesteps = timesteps.float()
    half = dim // 2
    if half == 0:
        return torch.zeros((timesteps.shape[0], dim), device=timesteps.device, dtype=timesteps.dtype)
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device, dtype=timesteps.dtype) / half)
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def sinusoidal_position_2d(h: int, w: int, dim: int, device, dtype):
    """
    2D sinusoidal position embedding of shape [1, h*w, dim].
    """
    assert dim >= 2, "pos dim must be >=2"
    dim_half = dim // 2
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    y = y.reshape(-1)
    x = x.reshape(-1)

    def _pe(coord, d):
        half = d // 2
        if half == 0:
            return torch.zeros((coord.shape[0], d), device=device, dtype=dtype)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device, dtype=dtype) / max(half, 1))
        args = coord[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if d % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    pe_x = _pe(x, dim_half)
    pe_y = _pe(y, dim - dim_half)
    pe = torch.cat([pe_x, pe_y], dim=-1)  # [h*w, dim]
    return pe.unsqueeze(0)  # [1, T, dim]


class AdaLNTransformerBlock(nn.Module):
    """
    Transformer block with AdaLayerNorm conditioning (time/cond).
    Operates on tokens [B, T, C].
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * dim), dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        # Produce scale and shift for two norms: [scale1, shift1, scale2, shift2] -> 4*C
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * dim)
        )

        # Zero-init last linear in MLP for stability (AdaLN-Zero)
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        """
        x: [B, T, C]; cond_vec: [B, cond_dim]
        """
        scale1, shift1, scale2, shift2 = self.ada(cond_vec).chunk(4, dim=-1)

        y = self.norm1(x)
        y = y * (1 + scale1[:, None, :]) + shift1[:, None, :]
        y = self.attn(y, y, y, need_weights=False)[0]
        x = x + y

        y = self.norm2(x)
        y = y * (1 + scale2[:, None, :]) + shift2[:, None, :]
        y = self.mlp(y)
        x = x + y
        return x


class TokenTransformerNoisePredictor(nn.Module):
    """
    Tiny DiT-style Transformer for noise prediction on very small spatial grids (H,W in {2,4,...}).

    Input:  x_t [B, C, H, W]
            t   [B] (long or float)
            cond: Optional[B, D_cond], will be fused into time embedding
    Output: eps_hat with the same shape as x_t
    """
    def __init__(
        self,
        in_channels: int,
        model_dim: Optional[int] = None,  # if None, use in_channels
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        cond_dim: Optional[int] = None,   # set if you pass cond; else None
        dropout: float = 0.0,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_dim = in_channels if model_dim is None else model_dim
        self.use_pos_emb = use_pos_emb
        self.time_embed_dim = time_embed_dim

        # Project channels to model dim (1x1 conv equals linear over channels)
        self.in_proj = nn.Conv2d(in_channels, self.model_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(self.model_dim, in_channels, kernel_size=1)

        # Time embedding to shared conditioning vector
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )

        # Optional condition embedding
        if cond_dim is not None:
            self.cond_proj = nn.Linear(cond_dim, self.model_dim)
        else:
            self.cond_proj = None

        # Stack of AdaLN Transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(dim=self.model_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  cond_dim=self.model_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(self.model_dim, eps=1e-6)

        # Initialize out_proj to small weights (predict small residual)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict noise eps with same shape as x.
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # 1) Channel projection
        h = self.in_proj(x)  # [B, D, H, W]
        D = h.shape[1]

        # 2) Tokens + optional 2D pos emb
        tokens = h.permute(0, 2, 3, 1).reshape(B, H * W, D)  # [B, T, D]
        if self.use_pos_emb:
            pos = sinusoidal_position_2d(H, W, D, device=x.device, dtype=x.dtype)  # [1, T, D]
            tokens = tokens + pos.to(dtype=tokens.dtype)

        # 3) Build conditioning vector from time (and optional cond)
        t_emb = timestep_embedding(t, dim=self.time_embed_dim).to(dtype=tokens.dtype)  # [B, time_embed_dim]
        t_ctx = self.time_mlp(t_emb)  # [B, D]
        if self.cond_proj is not None and cond is not None:
            t_ctx = t_ctx + self.cond_proj(cond.to(dtype=t_ctx.dtype))

        # 4) Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens, t_ctx)

        # 5) Back to image
        tokens = self.final_ln(tokens)
        h = tokens.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
        eps = self.out_proj(h)  # [B, C, H, W]
        return eps


# ===============================
# Demo
# ===============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    g = torch.Generator(device=device).manual_seed(42)

    # Example settings
    B, C = 2, 512
    sizes = [(2, 2), (4, 4)]  # try both 2x2 and 4x4
    time_embed_dim = 256
    cond_dim = 128  # if you want to use classifier-free guidance with a cond vector

    # Instantiate the tiny Transformer noise predictor
    model = TokenTransformerNoisePredictor(
        in_channels=C,
        model_dim=512,       # can be 256 to save memory
        num_layers=6,
        num_heads=8,
        mlp_ratio=4.0,
        time_embed_dim=time_embed_dim,
        cond_dim=cond_dim,   # set None if you will not pass cond
        dropout=0.0,
        use_pos_emb=True,
    ).to(device)

    model.eval()  # this is an untrained toy; for real use you should train it

    for (H, W) in sizes:
        # Optional: set up cond/uncond for classifier-free guidance
        use_cfg = False  # set True to demo CFG; note: untrained model, just for API/shape correctness
        if use_cfg:
            cond = torch.randn(B, cond_dim, device=device)
            uncond = torch.zeros(B, cond_dim, device=device)
            guidance_scale = 3.0
        else:
            cond = None
            uncond = None
            guidance_scale = 1.0

        x_final, x0_pred = ddim_sample(
            model=model,
            shape=(B, C, H, W),
            num_inference_steps=50,
            eta=0.0,                         # deterministic DDIM
            guidance_scale=guidance_scale,
            cond=cond,
            uncond_cond=uncond,
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
            device=device,
            dtype=torch.float32,
            generator=g,
            x_T=None,
        )

        print(f"HxW={H}x{W} -> Output x_final shape: {tuple(x_final.shape)}, x0_pred shape: {tuple(x0_pred.shape)}")
        # Expected: (B, C, H, W)