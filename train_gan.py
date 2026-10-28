import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from app.gan_model import Generator, Discriminator, LATENT_DIM

EPOCHS = 10
BATCH_SIZE = 128
LR = 0.0002

torch.manual_seed(42)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

# Scale MNIST pixels to [-1, 1] to match the generator's Tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
print(generator)
print(discriminator)

criterion = nn.BCELoss()  # discriminator already outputs a probability (Sigmoid)
opt_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

real_labels = torch.ones(BATCH_SIZE, 1, device=device)
fake_labels = torch.zeros(BATCH_SIZE, 1, device=device)

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()
    loss_D_sum, loss_G_sum = 0.0, 0.0
    progress = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_number, (real_images, _) in enumerate(progress):
        real_images = real_images.to(device)

        # 1) Train Discriminator: maximize log D(x) + log(1 - D(G(z)))
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        fake_images = generator(z)
        loss_D = criterion(discriminator(real_images), real_labels) + \
                 criterion(discriminator(fake_images.detach()), fake_labels)  # detach: don't update G here
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # 2) Train Generator: maximize log D(G(z)) (non-saturating loss, fake labelled as real)
        loss_G = criterion(discriminator(fake_images), real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loss_D_sum += loss_D.item()
        loss_G_sum += loss_G.item()
        if batch_number % 100 == 99:
            progress.set_postfix({
                "loss_D": f"{loss_D_sum/(batch_number+1):.4f}",
                "loss_G": f"{loss_G_sum/(batch_number+1):.4f}",
            })
    print(f"Epoch {epoch+1}: loss_D={loss_D_sum/len(train_loader):.4f} loss_G={loss_G_sum/len(train_loader):.4f}")

os.makedirs("models", exist_ok=True)
torch.save(generator.cpu().state_dict(), "models/gan_generator.pth")
print("Generator saved to models/gan_generator.pth")
