import io
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import os

# ---------------------------
# Device (CPU-only)
# ---------------------------
device = torch.device("cpu")

# ---------------------------
# Load Pix2Pix model (Generator only)
# ---------------------------
class SimplePix2PixGenerator(torch.nn.Module):
    # Minimal generator for demo (must match your trained model)
    def __init__(self):
        super().__init__()
        # Example: small UNet-like structure
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 3, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

@st.cache_resource(show_spinner=False)
def load_generator(model_path="./pix2pix_model/generator.pth"):
    gen = SimplePix2PixGenerator().to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()
    return gen

generator = load_generator()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎨 Sketch-to-Image Demo (CPU Pix2Pix)")
st.write("Upload a sketch (PNG/JPG) and generate a realistic image.")

uploaded = st.file_uploader("Upload sketch", type=["png", "jpg", "jpeg"])
if uploaded:
    sketch = Image.open(uploaded).convert("RGB")
    st.image(sketch, caption="Input Sketch", use_column_width=True)
else:
    sketch = None
    st.info("Please upload a sketch to continue.")

# ---------------------------
# Generate image button
# ---------------------------
if st.button("Generate Image") and sketch:
    try:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        input_tensor = transform(sketch).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = generator(input_tensor)
            output_tensor = (output_tensor + 1) / 2  # scale to [0,1]

        # Convert to PIL Image
        output_image = T.ToPILImage()(output_tensor.squeeze(0).cpu())
        st.success("✅ Generation complete!")
        st.image(output_image, caption="Generated Image", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Image", data=buf, file_name="generated.png", mime="image/png")

    e
