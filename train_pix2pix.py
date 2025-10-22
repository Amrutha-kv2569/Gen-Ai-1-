import os
import zipfile
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Unzip dataset if needed
# ---------------------------
if not os.path.exists("dataset"):
    if os.path.exists("dataset.zip"):
        with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ dataset.zip extracted successfully!")
    else:
        raise FileNotFoundError("dataset.zip not found in repo!")

# ---------------------------
# Dataset Loader
# ---------------------------
class SketchDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, transform=None):
        self.sketches = sorted(os.listdir(sketch_dir))
        self.photos = sorted(os.listdir(photo_dir))
        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir
        self.transform = transform

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch = Image.open(os.path.join(self.sketch_dir, self.sketches[idx])).convert("RGB")
        photo = Image.open(os.path.join(self.photo_dir, self.photos[idx])).convert("RGB")
        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)
        return sketch, photo

# ---------------------------
# Generator (U-Net style)
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ---------------------------
# Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128*16*16, 1),
            nn.Sigmoid()
        )

    def forward(self, sketch, photo):
        x = torch.cat([sketch, photo], dim=1)
        return self.main(x)

# ---------------------------
# Setup
# ---------------------------
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = SketchDataset("dataset/sketches", "dataset/photos", transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# ---------------------------
# Train
# ---------------------------
epochs = 5  # tiny dataset, just demo
for epoch in range(epochs):
    for sketch, photo in loader:
        sketch, photo = sketch.to(device), photo.to(device)
        valid = torch.ones((1, 1), device=device)
        fake = torch.zeros((1, 1), device=device)

        # Train Generator
        optimizer_G.zero_grad()
        gen_photo = G(sketch)
        D_gen = D(sketch, gen_photo)
        loss_G = criterion(D_gen, valid)
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        D_real = D(sketch, photo)
        D_fake = D(sketch, gen_photo.detach())
        loss_D = (criterion(D_real, valid) + criterion(D_fake, fake)) / 2
        loss_D.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss_G: {loss_G.item():.4f}  Loss_D: {loss_D.item():.4f}")

# ---------------------------
# Save Generator weights
# ---------------------------
torch.save(G.state_dict(), "pix2pix_generator.pth")
print("✅ Training complete! Generator saved as pix2pix_generator.pth")
