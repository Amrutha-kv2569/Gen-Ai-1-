import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# ---------------------------
# Autoencoder (same as training)
# ---------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    model = Autoencoder()
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    try:
        model.load_state_dict(torch.load("autoencoder_model.pth", map_location=device, weights_only=False))
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ Model not found. Please run train_autoencoder.py first.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
    return model

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🖌 Sketch-to-Image (Autoencoder)")
st.write("Upload a black-and-white sketch and generate a colored image.")

uploaded_file = st.file_uploader("Upload Sketch (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file and model is not None:
    sketch_img = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    sketch_tensor = transform(sketch_img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output_tensor = model(sketch_tensor)

    # Convert tensor to image
    output_img = output_tensor.squeeze(0).permute(1,2,0).numpy()
    output_img = (output_img * 255).astype('uint8')
    output_img = Image.fromarray(output_img)

    st.image([sketch_img, output_img], caption=["Input Sketch", "Generated Image"], width=300)
