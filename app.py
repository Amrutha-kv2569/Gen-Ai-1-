import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn

# ---------------------------
# Generator (same as training)
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
# Load model safely
# ---------------------------
@st.cache_resource
def load_model():
    model = Generator()
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    try:
        # Fix PyTorch 2.6+ unpickling error
        model.load_state_dict(torch.load("pix2pix_generator.pth", map_location=device, weights_only=False))
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ Model not found. Please train it using train_pix2pix.py first.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

    return model

G = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🖌 Sketch-to-Image Generation")
st.write("Upload a hand-drawn or digital sketch and convert it into a realistic image using Pix2Pix.")

uploaded_file = st.file_uploader("Upload your sketch image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file and G is not None:
    # Open image
    sketch_img = Image.open(uploaded_file).convert("RGB")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    sketch_tensor = transform(sketch_img).unsqueeze(0)  # Add batch dimension

    # Generate
    with torch.no_grad():
        output_tensor = G(sketch_tensor)
    
    # Denormalize
    output_tensor = (output_tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    output_img = transforms.ToPILImage()(output_tensor)

    # Show
    st.image([sketch_img, output_img], caption=["Input Sketch", "Generated Image"], width=300)
