import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from app.diffusion_model import UNet, diffusion_loss

EPOCHS = 30
BATCH_SIZE = 128
LR = 2e-4

torch.manual_seed(42)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

# CIFAR-10 at native 32x32, pixels scaled to [-1, 1] (same range as the Gaussian noise)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = UNet().to(device)
print(f"UNet parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0.0
    progress = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_number, (images, _) in enumerate(progress):
        loss = diffusion_loss(model, images.to(device))  # labels unused: unconditional generation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if batch_number % 100 == 99:
            progress.set_postfix({"loss": f"{loss_sum/(batch_number+1):.4f}"})
    print(f"Epoch {epoch+1}: loss={loss_sum/len(train_loader):.4f}")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/diffusion_unet.pth")  # save every epoch so you can stop early

print("UNet saved to models/diffusion_unet.pth")
