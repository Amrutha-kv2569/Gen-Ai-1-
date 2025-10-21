import io
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ---------------------------
# Simple Pix2Pix Generator (example small model)
# ---------------------------
class TinyGenerator(nn.Module):
    def __init__(self):
        super(TinyGenerator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------------
# Load trained model
# ---------------------------
device = torch.device("cpu")
model = TinyGenerator().to(device)
model.load_state_dict(torch.load("tiny_pix2pix.pth", map_location=device))
model.eval()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎨 Sketch-to-Image (Tiny Pix2Pix CPU)")
st.write("Upload a sketch and generate a photorealistic image using a tiny Pix2Pix model on CPU.")

uploaded = st.file_uploader("Upload sketch (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    sketch = Image.open(uploaded).convert("RGB")
    st.image(sketch, caption="Input Sketch", use_column_width=True)
else:
    sketch = None
    st.info("Please upload a sketch to continue.")

# ---------------------------
# Transform sketch for model
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),  # small size for CPU
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

if sketch and st.button("Generate Image"):
    try:
        input_tensor = transform(sketch).unsqueeze(0).to(device)  # add batch dim
        with torch.no_grad():
            output = model(input_tensor).squeeze(0).cpu()
        
        # Convert back to PIL
        output_image = output * 0.5 + 0.5  # unnormalize
        output_image = transforms.ToPILImage()(output_image.clamp(0,1))
        st.success("✅ Generation complete!")
        st.image(output_image, caption="Generated Image", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Image", data=buf, file_name="pix2pix_result.png", mime="image/png")

    except Exception as e:
        st.error(f"Failed to generate image: {e}")
