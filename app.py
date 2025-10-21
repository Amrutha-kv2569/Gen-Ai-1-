import os
import io
import random
import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ---------------------------
# Load pipeline
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(model_id="timbrooks/instruct-pix2pix"):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=torch.float16 if device.type=="cuda" else torch.float32
    )
    pipe.to(device)
    return pipe

st.title("🎨 Sketch-to-Image using Pix2Pix")
st.write("Upload a hand-drawn or digital sketch and convert it into a photorealistic image.")

pipe = load_pipeline()

# ---------------------------
# Upload sketch
# ---------------------------
uploaded = st.file_uploader("Upload sketch (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    sketch = Image.open(uploaded).convert("RGB")
    st.image(sketch, caption="Input Sketch", use_column_width=True)
else:
    st.info("Upload a sketch to continue.")

# ---------------------------
# Prompt and parameters
# ---------------------------
prompt = st.text_area("Prompt", value="A realistic painting of a mountain landscape with lake")
negative_prompt = st.text_area("Negative prompt", value="low quality, blurry, deformed", height=60)
num_steps = st.slider("Steps", 10, 100, 28)
guidance = st.slider("Guidance scale", 1.0, 20.0, 7.5)
seed_val = st.number_input("Seed (0 for random)", min_value=0, value=0, step=1)

# ---------------------------
# Generate
# ---------------------------
if st.button("Generate Image"):
    if not uploaded:
        st.warning("Please upload a sketch first.")
    else:
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

        st.success("Generation complete!")
        st.image(result, caption="Generated Image", use_column_width=True)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download result", data=buf, file_name="pix2pix_result.png", mime="image/png")
