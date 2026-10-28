import io
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from PIL import Image

LATENT_DIM = 100
MODEL_PATH = "models/gan_generator.pth"


class Generator(nn.Module):
    """z (B, 100) -> FC -> (B, 128, 7, 7) -> ConvT -> (B, 64, 14, 14) -> ConvT -> (B, 1, 28, 28)"""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 14x14 -> 28x28
            nn.Tanh(),  # pixel values in [-1, 1], matching normalized MNIST
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)  # reshape (B, 6272) -> (B, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    """(B, 1, 28, 28) -> Conv -> (B, 64, 14, 14) -> Conv -> (B, 128, 7, 7) -> Linear -> (B, 1) probability"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 28x28 -> 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # real/fake probability
        )

    def forward(self, x):
        return self.net(x)


_generator = None


def _load_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
        _generator.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _generator.eval()
    return _generator


def generate_digits_png(num_images: int = 16) -> bytes:
    """Sample num_images digits from the trained generator and return them as a PNG grid."""
    generator = _load_generator()
    with torch.no_grad():
        images = generator(torch.randn(num_images, LATENT_DIM))  # (N, 1, 28, 28) in [-1, 1]
    grid = make_grid(images, nrow=min(num_images, 8), normalize=True, value_range=(-1, 1), pad_value=1)
    array = (grid[0] * 255).byte().numpy()
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()
