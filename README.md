# sps_genai

A FastAPI project that serves several generative AI / deep learning models:
- Bigram text generator
- Word embeddings (spaCy)
- CNN image classifier trained on CIFAR-10 (Assignment 2)
- GAN hand-written digit generator trained on MNIST (Assignment 3)
- Diffusion model and Energy-Based Model image generators trained on CIFAR-10 (Assignment 4)

## Setup
```bash
uv sync
```

## Run the API
```bash
uv run fastapi dev app/main.py
```
Interactive docs: http://127.0.0.1:8000/docs

## Endpoints
- `POST /generate` — JSON body `{"start_word": "bigram", "length": 5}`
- `POST /embedding` — JSON body `{"word": "apple"}`
- `POST /classify` — upload an image file (multipart/form-data, field name `file`).
  Returns the predicted CIFAR-10 class and its confidence, e.g.
  `{"filename": "cat.jpg", "class": "cat", "confidence": 0.8123}`

  Example with curl:
```bash
  curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/classify
```
- `POST /generate_digits` — JSON body `{"num_images": 16}` (1–64). Returns a PNG grid of
  GAN-generated hand-written digits.
```bash
  curl -X POST -H "Content-Type: application/json" -d '{"num_images": 16}' \
       http://127.0.0.1:8000/generate_digits -o digits.png
```
- `POST /generate_diffusion` — JSON body `{"num_images": 16, "num_steps": 50}`. Returns a PNG grid of
  CIFAR-10-style images sampled from the diffusion model (`num_steps` = DDIM sampling steps, 1–1000).
```bash
  curl -X POST -H "Content-Type: application/json" -d '{"num_images": 16, "num_steps": 50}' \
       http://127.0.0.1:8000/generate_diffusion -o diffusion.png
```
- `POST /generate_ebm` — JSON body `{"num_images": 16, "steps": 256}`. Returns a PNG grid of images
  sampled from the energy model with Langevin dynamics (`steps` = number of Langevin steps, 1–2000).
```bash
  curl -X POST -H "Content-Type: application/json" -d '{"num_images": 16, "steps": 256}' \
       http://127.0.0.1:8000/generate_ebm -o ebm.png
```

## CNN Classifier (Assignment 2)

### Architecture (`app/cnn_model.py`)
Input 64×64×3 → Conv(16, 3×3, stride 1, pad 1) → ReLU → MaxPool(2×2, stride 2)
→ Conv(32, 3×3, stride 1, pad 1) → ReLU → MaxPool(2×2, stride 2)
→ Flatten → FC(100) → ReLU → FC(10)

### Training (`train_cnn.py`)
CIFAR-10 images (32×32) are resized to 64×64 to match the required input size.
Trained for 10 epochs with Adam (lr = 0.0005), batch size 32.
Test accuracy: **66.09%**.

The trained weights are included at `models/cnn_cifar10.pth`, so retraining is not required.
To retrain:
```bash
uv run python train_cnn.py
```

## GAN Digit Generator (Assignment 3)

### Architecture (`app/gan_model.py`)
**Generator:** noise (B, 100) → FC → 7×7×128 → reshape
→ ConvTranspose(128→64, 4×4, stride 2, pad 1) → BatchNorm → ReLU (14×14)
→ ConvTranspose(64→1, 4×4, stride 2, pad 1) → Tanh (1×28×28)

**Discriminator:** (1, 28, 28) → Conv(1→64, 4×4, stride 2, pad 1) → LeakyReLU(0.2) (14×14)
→ Conv(64→128, 4×4, stride 2, pad 1) → BatchNorm → LeakyReLU(0.2) (7×7)
→ Flatten → Linear(6272→1) → Sigmoid (real/fake probability)

### Training (`train_gan.py`)
MNIST pixels are normalized to [-1, 1] to match the Tanh output. Trained for 10 epochs
with BCE loss and Adam (lr = 0.0002, betas = (0.5, 0.999)), batch size 128.
The trained generator weights are included at `models/gan_generator.pth`. To retrain:
```bash
uv run python train_gan.py
```

## Diffusion Model (Assignment 4)

### Architecture (`app/diffusion_model.py`)
- **Time embedding:** sinusoidal, `emb = [sin(t·f_i), cos(t·f_i)]` with `f_i = 10000^(-i/64)` (dim 128),
  followed by Linear(128→256) → SiLU → Linear(256→256). Each ResBlock adds a projection of it to its feature map.
- **UNet** (3.89M params): 3×32×32 → Conv(3→64) → ResBlock(64) [skip 1] → stride-2 Conv (16×16)
  → ResBlock(64→128) [skip 2] → stride-2 Conv (8×8) → ResBlock(128→256) → ResBlock(256)
  → Upsample+Conv(256→128), concat skip 2 → ResBlock(256→128) (16×16)
  → Upsample+Conv(128→64), concat skip 1 → ResBlock(128→64) (32×32) → GroupNorm → SiLU → Conv(64→3).
  Output = predicted noise ε.
- **Forward process:** linear β schedule (1e-4 → 0.02, T = 1000), `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`.
- **Loss:** `MSE(ε, ε_θ(x_t, t))` with a random t per image.
- **Sampling:** deterministic DDIM (η = 0) on `num_steps` evenly spaced timesteps (default 50).

### Training (`train_diffusion.py`)
CIFAR-10 (32×32, random horizontal flip), pixels scaled to [-1, 1]. Adam (lr = 2e-4), batch size 128,
30 epochs. Weights are saved after every epoch to `models/diffusion_unet.pth`.
```bash
uv run python train_diffusion.py
```

## Energy-Based Model (Assignment 4)

### Architecture (`app/energy_model.py`)
3×32×32 → 4 × [Conv(3×3, stride 2) → SiLU] with channels 32/64/128/128 (32→16→8→4→2)
→ Flatten → Linear(512→64) → SiLU → Linear(64→1) = energy E(x). Low energy = realistic image.
SiLU (Swish) is used because it is smooth, so ∂E/∂x is well behaved during Langevin sampling.

### Langevin sampling
Gradient descent on the **input image** rather than on the weights:
`x ← clip(x − step_size · clip(∂E/∂x, ±0.03) + noise)`. Inside the sampler the model's parameters are set to
`requires_grad=False` and `x` to `requires_grad=True`; `torch.autograd.grad(E(x).sum(), x)` gives ∂E/∂x.
Defaults: step size 10, noise std 0.005.

### Training (`train_ebm.py`)
Contrastive divergence: `loss = E(real) − E(fake) + 0.1·(E(real)² + E(fake)²)`. Fake images come from
60 Langevin steps started from a replay buffer (95% previous samples, 5% fresh uniform noise).
Adam (lr = 1e-4, betas = (0.0, 0.999)), gradient-norm clipping 0.1, batch size 128, 10 epochs.
```bash
uv run python train_ebm.py
```
At inference time the API starts from uniform noise and runs `steps` Langevin steps (default 256).

### Project structure
```
app/
  main.py          # FastAPI app and endpoints
  bigram_model.py  # bigram text generator
  embeddings.py    # spaCy word embeddings
  cnn_model.py     # CNN architecture definition
  classifier.py    # loads trained weights and runs prediction
  gan_model.py     # GAN generator/discriminator + digit sampling
  diffusion_model.py # UNet, noise schedule, diffusion loss, DDIM sampling
  energy_model.py  # energy network, Langevin sampling, CD loss, replay buffer
  image_utils.py   # tensor -> PNG grid helper
models/
  cnn_cifar10.pth  # trained CNN weights
  gan_generator.pth # trained GAN generator weights
  diffusion_unet.pth # trained diffusion UNet weights
  ebm_cifar10.pth  # trained energy model weights
train_cnn.py       # CNN training script
train_gan.py       # GAN training script
train_diffusion.py # diffusion model training script
train_ebm.py       # energy model training script
```