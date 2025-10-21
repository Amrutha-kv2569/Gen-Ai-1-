import io
import os
import requests
from PIL import Image
import torch
import streamlit as st
from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# ---------------------------
# CPU device
# ---------------------------
device = torch.device("cpu")

# ---------------------------
# Load Pix2Pix pipeline
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(model_id="timbrooks/instruct-pix2pix"):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=torch.float32
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipeline()

st.title("🎨 CPU Pix2Pix Sketch-to-Image")
st.write("Select a sketch from the dataset or upload your own.")

# ---------------------------
# GitHub dataset
# ---------------------------
# Replace this URL with your GitHub raw folder containing sketches
GITHUB_RAW_URL = "https://github.com/Amrutha-kv2569/Gen-Ai-1-/blob/main/dataset.zip"

# Example images: list of filenames in your repo
image_files = ["n02691156_58", "n02691156_196", "n02691156_394"]

selected_image_name = st.selectbox("Select a sketch from dataset", ["--Upload manually--"] + image_files)

if selected_image_name != "--Upload manually--":
    # Load image from GitHub
    url = GITHUB_RAW_URL + selected_image_name
    response = requests.get(url)
    sketch = Image.open(io.BytesIO(response.content)).convert("RGB")
    st.image(sketch, caption=f"Selected sketch: {selected_image_name}", use_column_width=True)
else:
    uploaded = st.file_uploader("Upload sketch (PNG/JPG)", type=["png","jpg","jpeg"])
    if uploaded:
        sketch = Image.open(uploaded).convert("RGB")
        st.image(sketch, caption="Uploaded sketch", use_column_width=True)
    else:
        st.info("Select a sketch or upload one to continue.")
        sketch = None

# ---------------------------
# Prompt and parameters
# ---------------------------
prompt = st.text_area("Prompt", value="A realistic painting of a mountain landscape")
negative_prompt = st.text_area("Negative prompt", value="blurry, low quality", height=60)
num_steps = st.slider("Steps", 10, 50, 25)
guidance = st.slider("Guidance scale", 1.0, 20.0, 7.5)
seed_val = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)

# ---------------------------
# Generate
# ---------------------------
if st.button("Generate Image") and sketch:
    seed = None if seed_val == 0 else int(seed_val)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    with torch.autocast(device.type if device.type != "cpu" else "cpu"):
        result = pipe(
            image=sketch,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]

    st.success("✅ Generation complete!")
    st.image(result, caption="Generated Image", use_column_width=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download Image", data=buf, file_name="pix2pix_result.png", mime="image/png")
