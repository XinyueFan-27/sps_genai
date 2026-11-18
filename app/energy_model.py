import os
import torch
import torch.nn as nn
from app.image_utils import rgb_grid_png

MODEL_PATH = "models/ebm_cifar10.pth"


class EnergyModel(nn.Module):
    """(B, 3, 32, 32) image -> (B,) scalar energy E(x). Real images should get LOW energy."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.SiLU(),     # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),    # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),   # 8x8 -> 4x4
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.SiLU(),  # 4x4 -> 2x2
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        # SiLU (Swish) is smooth, so dE/dx is well-behaved for Langevin dynamics (ReLU's gradient is not)

    def forward(self, x):
        return self.net(x).squeeze(1)


def langevin_sample(model, x, steps: int = 60, step_size: float = 10.0, noise_std: float = 0.005, grad_clip: float = 0.03):
    """
    Langevin dynamics: gradient DESCENT on the INPUT image to lower its energy.
        x <- x - step_size * dE/dx + noise
    The model's weights are frozen here; only x tracks gradients.
    """
    was_training = model.training
    model.eval()
    # freeze parameters: we don't want gradients w.r.t. the weights during sampling
    param_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    x = x.clone().detach()
    with torch.enable_grad():                      # works even if the caller is inside torch.no_grad()
        for _ in range(steps):
            x = (x + noise_std * torch.randn_like(x)).clamp(-1, 1)  # small noise keeps the chain exploring
            x.requires_grad_(True)                  # track gradients w.r.t. the image
            energy = model(x).sum()                 # sum over batch -> scalar; grads stay per-image
            (grad_x,) = torch.autograd.grad(energy, x)  # dE/dx, shape (B, 3, 32, 32)
            grad_x = grad_x.clamp(-grad_clip, grad_clip)  # stability: limit each pixel's update
            x = (x - step_size * grad_x).clamp(-1, 1).detach()  # descent step, then cut the graph

    for p, flag in zip(model.parameters(), param_flags):
        p.requires_grad_(flag)
    model.train(was_training)
    return x


def contrastive_divergence_loss(model, real, fake, alpha: float = 0.1):
    """Push energy of real data down, energy of sampled (fake) data up; alpha term keeps energies near 0."""
    e_real, e_fake = model(real), model(fake)
    cd_loss = e_real.mean() - e_fake.mean()
    reg_loss = alpha * (e_real.pow(2).mean() + e_fake.pow(2).mean())
    return cd_loss + reg_loss, cd_loss, reg_loss


class ReplayBuffer:
    """Keeps past Langevin samples; each batch starts 95% from old samples, 5% from fresh uniform noise."""

    def __init__(self, max_size: int = 8192):
        self.max_size = max_size
        self.examples = [torch.rand(128, 3, 32, 32) * 2 - 1]

    def sample(self, model, batch_size: int, steps: int = 60, device="cpu"):
        n_new = max(1, int(0.05 * batch_size))
        old = torch.cat(self.examples, dim=0)
        old = old[torch.randint(0, old.shape[0], (batch_size - n_new,))]
        start = torch.cat([torch.rand(n_new, 3, 32, 32) * 2 - 1, old], dim=0).to(device)
        fake = langevin_sample(model, start, steps=steps)
        self.examples = [fake.cpu()] + self.examples
        self.examples = list(torch.cat(self.examples, 0)[: self.max_size].split(batch_size))
        return fake


_model = None


def _load_model() -> EnergyModel:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found - run `uv run python train_ebm.py` first")
        _model = EnergyModel()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
    return _model


def generate_ebm_png(num_images: int = 16, steps: int = 256) -> bytes:
    """Start from uniform noise in [-1, 1] and run Langevin dynamics on the trained energy function."""
    x = torch.rand(num_images, 3, 32, 32) * 2 - 1
    images = langevin_sample(_load_model(), x, steps=steps)
    return rgb_grid_png(images, nrow=min(num_images, 8))
