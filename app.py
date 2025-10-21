import io
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import os

# ---------------------------
# Device (CPU-only)
# ---------------------------
device = torch.device("cpu")

# ---------------------------
# Define the generator (must match your trained model)
# ---------------------------
class SimplePix2PixGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 3, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------------
# Load generator from local file
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_generator(model_path="./pix2pix_model/generator.pth"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please upload generator.pth in pix2pix_model/")
        st.stop()
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
        # Transform sketch to tensor
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        input_tensor = transform(sketch).unsqueeze(0).to(device)

        # Generate output image
        with torch.no_grad():
            output_tensor = generator(input_tensor)
            output_tensor = (output_tensor + 1) / 2  # scale to [0,1]

        # Convert tensor to PIL image
        output_image = T.ToPILImage()(output_tensor.squeeze(0).cpu())

        # Display generated image
        st.success("✅ Generation complete!")
        st.image(output_image, caption="Generated Image", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Image", data=buf, file_name="generated.png", mime="image/png")

    except Exception as e:
        st.error(f"Failed to generate image: {e}")
