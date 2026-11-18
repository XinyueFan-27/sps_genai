import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from app.energy_model import EnergyModel, ReplayBuffer, contrastive_divergence_loss

EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-4
LANGEVIN_STEPS = 60

torch.manual_seed(42)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1], same range as Langevin samples
])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = EnergyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.0, 0.999))
buffer = ReplayBuffer()

for epoch in range(EPOCHS):
    model.train()
    cd_sum, reg_sum = 0.0, 0.0
    progress = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_number, (real_images, _) in enumerate(progress):
        real_images = real_images.to(device)
        real_images = (real_images + 0.005 * torch.randn_like(real_images)).clamp(-1, 1)  # small noise on data

        # 1) Negative samples: Langevin dynamics on the INPUT (weights frozen inside langevin_sample)
        fake_images = buffer.sample(model, BATCH_SIZE, steps=LANGEVIN_STEPS, device=device)

        # 2) Update the weights: lower E(real), raise E(fake)
        loss, cd_loss, reg_loss = contrastive_divergence_loss(model, real_images, fake_images)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        cd_sum += cd_loss.item()
        reg_sum += reg_loss.item()
        if batch_number % 50 == 49:
            progress.set_postfix({"cd": f"{cd_sum/(batch_number+1):.4f}", "reg": f"{reg_sum/(batch_number+1):.4f}"})
    print(f"Epoch {epoch+1}: cd_loss={cd_sum/len(train_loader):.4f} reg_loss={reg_sum/len(train_loader):.4f}")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ebm_cifar10.pth")

print("Energy model saved to models/ebm_cifar10.pth")
