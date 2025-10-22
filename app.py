import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# --- Generator Model (must match training) ---
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

# --- Load model ---
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("pix2pix_generator.pth", map_location="cpu"))
    model.eval()
    return model

G = load_model()

# --- UI ---
st.title("🖍️ Sketch to Image Generator (CPU Pix2Pix)")
st.write("Upload a hand-drawn sketch to generate a realistic image (trained on your airplane dataset).")

uploaded = st.file_uploader("Upload a sketch", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Input Sketch", width=256)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor_img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = G(tensor_img).squeeze(0)
        output = (output * 0.5 + 0.5).clamp(0, 1)
        result_img = transforms.ToPILImage()(output)

    st.image(result_img, caption="Generated Image", width=256)

    # Download button
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    st.download_button("Download Image", data=buf.getvalue(), file_name="generated.png", mime="image/png")
