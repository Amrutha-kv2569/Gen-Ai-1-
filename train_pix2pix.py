import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os  # <-- make sure this is imported
import zipfile
import os

# Automatically extract dataset.zip if the dataset folder doesn't exist
if not os.path.exists("dataset"):
    if os.path.exists("dataset.zip"):
        print("📦 Extracting dataset.zip ...")
        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall("dataset")
        print("✅ Dataset extracted to /dataset")
    else:
        raise FileNotFoundError("❌ dataset.zip not found. Please upload it to your project root.")


# ---------------------------
# Check dataset folders
# ---------------------------
sketch_dir = "dataset/sketches"
photo_dir = "dataset/photos"

if not os.path.exists(sketch_dir) or not os.path.exists(photo_dir):
    raise FileNotFoundError(f"Dataset folders not found. Make sure '{sketch_dir}' and '{photo_dir}' exist.")

# ---------------------------
# Dataset class
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
