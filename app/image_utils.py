import io
from PIL import Image
from torchvision.utils import make_grid


def rgb_grid_png(images, nrow: int = 8) -> bytes:
    """(N, 3, H, W) tensor in [-1, 1] -> PNG grid bytes."""
    grid = make_grid(images.cpu(), nrow=nrow, normalize=True, value_range=(-1, 1), pad_value=1)
    array = (grid.permute(1, 2, 0) * 255).byte().numpy()  # (H, W, 3)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
