import os
import zipfile
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st
from train_pix2pix import SimplePix2Pix

# --- Step 1: Unzip dataset if not already ---
if not os.path.exists("dataset"):
    if os.path.exists("dataset.zip"):
        with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        st.info("✅ Dataset extracted successfully.")
    else:
        st.error("❌ 'dataset.zip' not found! Please upload it to the project folder.")
        st.stop()

# --- Step 2: Define model loading function ---
@st.cache_resource
def load_model():
    model = SimplePix2Pix()
    if os.path.exists("pix2pix_model.pth"):
        model.load_state_dict(torch.load("pix2pix_model.pth", map_location=torch.device("cpu")))
        st.success("Model loaded successfully!")
    else:
        st.warning("⚠️ No pre-trained model found. Using untrained model for demo.")
    model.eval()
    return model

model = load_model()

# --- Step 3: Streamlit UI ---
st.title("✏️ Sketch to Image Generator (CPU Demo)")
st.write("Upload a sketch and generate a simple image using Pix2Pix model.")

uploaded_file = st.file_uploader("Upload a sketch image", type=["jpg", "png"])

if uploaded_file is not None:
    sketch = Image.open(uploaded_file).convert("RGB")
    st.image(sketch, caption="🖼️ Uploaded Sketch", use_container_width=True)

    # --- Preprocess input ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    sketch_tensor = transform(sketch).unsqueeze(0)

    # --- Generate output ---
    with torch.no_grad():
        output = model(sketch_tensor)
    output_image = output.squeeze().permute(1, 2, 0).detach().numpy()
    output_image = (output_image + 1) / 2  # Denormalize to [0,1]

    st.image(output_image, caption="🖼️ Generated Image", use_container_width=True)
