import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.image_utils import rgb_grid_png

MODEL_PATH = "models/diffusion_unet.pth"
T = 1000           # number of diffusion timesteps
TIME_DIM = 128     # size of the sinusoidal time embedding


# ---------------------------------------------------------------------------
# 1) Sinusoidal time embedding
#    emb[2i]   = sin(t / 10000^(i / half))
#    emb[2i+1] = cos(t / 10000^(i / half)),   i = 0 .. half-1,  half = dim / 2
#    (we put all sin values first, then all cos values: [sin..., cos...])
# ---------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int = TIME_DIM, max_period: float = 10000.0) -> torch.Tensor:
    """t: (B,) integer timesteps -> (B, dim)"""
    half = dim // 2
    i = torch.arange(half, device=t.device, dtype=torch.float32)
    freqs = torch.exp(-math.log(max_period) * i / half)       # 1 / 10000^(i/half), (half,)
    args = t.float()[:, None] * freqs[None, :]                # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


# ---------------------------------------------------------------------------
# 2) UNet that predicts the noise eps added to x_t, conditioned on t
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Two 3x3 convs; the time embedding is projected and added in between."""

    def __init__(self, in_c, out_c, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.time_proj = nn.Linear(t_dim, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]  # broadcast (B, C) over H, W
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """(B, 3, 32, 32) noisy image + (B,) timestep -> (B, 3, 32, 32) predicted noise"""

    def __init__(self, t_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inc = nn.Conv2d(3, 64, 3, padding=1)
        # encoder: stride-2 convs halve the resolution 32 -> 16 -> 8
        self.down1 = ResBlock(64, 64, t_dim)
        self.pool1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.down2 = ResBlock(64, 128, t_dim)
        self.pool2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        # bottleneck at 8x8
        self.mid1 = ResBlock(128, 256, t_dim)
        self.mid2 = ResBlock(256, 256, t_dim)
        # decoder: upsample, concat the matching skip connection, ResBlock
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, padding=1))
        self.dec2 = ResBlock(256, 128, t_dim)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1))
        self.dec1 = ResBlock(128, 64, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_embedding(t))            # (B, t_dim)
        x = self.inc(x)                                           # (B, 64, 32, 32)
        s1 = self.down1(x, t_emb)                                 # (B, 64, 32, 32)
        s2 = self.down2(self.pool1(s1), t_emb)                    # (B, 128, 16, 16)
        h = self.pool2(s2)                                        # (B, 128, 8, 8)
        h = self.mid2(self.mid1(h, t_emb), t_emb)                 # (B, 256, 8, 8)
        h = self.dec2(torch.cat([self.up2(h), s2], 1), t_emb)     # (B, 128, 16, 16)
        h = self.dec1(torch.cat([self.up1(h), s1], 1), t_emb)     # (B, 64, 32, 32)
        return self.out(h)                                        # (B, 3, 32, 32)


# ---------------------------------------------------------------------------
# 3) Noise schedule, forward process and training loss (DDPM)
# ---------------------------------------------------------------------------
betas = torch.linspace(1e-4, 0.02, T)          # beta_t, linear schedule
alpha_bars = torch.cumprod(1.0 - betas, dim=0)  # alpha_bar_t = prod_{s<=t} (1 - beta_s)


def q_sample(x0, t, noise):
    """Forward process in one jump: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps"""
    ab = alpha_bars.to(x0.device)[t][:, None, None, None]  # (B, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def diffusion_loss(model, x0):
    """Pick random t, add noise, ask the UNet to predict that noise: L = ||eps - eps_theta(x_t, t)||^2"""
    t = torch.randint(0, T, (x0.shape[0],), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    return F.mse_loss(model(x_t, t), noise)


# ---------------------------------------------------------------------------
# 4) Reverse process: DDIM sampling (deterministic, eta = 0)
#    Uses a subset of `num_steps` timesteps, so 50 steps instead of 1000 is fine for an API.
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(model, num_images: int, num_steps: int = 50, device="cpu"):
    ab = alpha_bars.to(device)
    x = torch.randn(num_images, 3, 32, 32, device=device)          # start from pure noise x_T
    timesteps = torch.linspace(T - 1, 0, num_steps, device=device).long()
    for i, t in enumerate(timesteps):
        ab_t = ab[t]
        ab_prev = ab[timesteps[i + 1]] if i + 1 < num_steps else torch.tensor(1.0, device=device)
        eps = model(x, t.repeat(num_images))                          # predicted noise
        x0_pred = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)  # invert q_sample
        x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps     # jump to the previous timestep
    return x  # (N, 3, 32, 32) in [-1, 1]


_model = None


def _load_model() -> UNet:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found - run `uv run python train_diffusion.py` first")
        _model = UNet()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
    return _model


def generate_diffusion_png(num_images: int = 16, num_steps: int = 50) -> bytes:
    images = sample(_load_model(), num_images, num_steps)
    return rgb_grid_png(images, nrow=min(num_images, 8))
