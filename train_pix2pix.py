import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# ---------------------------
# Dataset
# ---------------------------
class SketchPhotoDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir
        self.transform = transform
        self.files = sorted(os.listdir(sketch_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.files[idx])
        photo_path = os.path.join(self.photo_dir, self.files[idx])

        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)

        return sketch, photo

# ---------------------------
# Transform (resize for CPU)
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = SketchPhotoDataset("dataset/sketches", "dataset/photos", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ---------------------------
# Simple Pix2Pix Generator (U-Net style)
# ---------------------------
class SimplePix2Pix(nn.Module):
    def __init__(self):
        super(SimplePix2Pix, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cpu")
model = SimplePix2Pix().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ---------------------------
# Training
# ---------------------------
epochs = 50  # small dataset, small model
for epoch in range(epochs):
    for sketch, photo in dataloader:
        sketch = sketch.to(device)
        photo = photo.to(device)

        output = model(sketch)
        loss = criterion(output, photo)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "pix2pix_model.pth")
print("Training complete. Model saved as pix2pix_model.pth")
