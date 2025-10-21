import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import io
from train_pix2pix import SimplePix2Pix  # import model class

# ---------------------------
# Load trained model
# ---------------------------
device = torch.device("cpu")
model = SimplePix2Pix().to(device)
model.load_state_dict(torch.load("pix2pix_model.pth", map_location=device))
model.eval()

# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎨 Sketch-to-Image (CPU Pix2Pix)")
st.write("Upload a sketch to generate a realistic image.")

uploaded = st.file_uploader("Upload sketch", type=["png","jpg","jpeg"])
if uploaded:
    sketch = Image.open(uploaded).convert("RGB")
    st.image(sketch, caption="Input Sketch", use_column_width=True)

    # Preprocess
    input_tensor = transform(sketch).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess
    output_tensor = (output_tensor.squeeze(0).clamp(0,1) * 255).byte()
    output_img = transforms.ToPILImage()(output_tensor.cpu())
    st.image(output_img, caption="Generated Image", use_column_width=True)

    # Download button
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download Image", data=buf, file_name="generated.png", mime="image/png")
else:
    st.info("Please upload a sketch to continue.")
