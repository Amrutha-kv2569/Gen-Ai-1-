import io
import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# ---------------------------
# CPU device
# ---------------------------
device = torch.device("cpu")

# ---------------------------
# Load lightweight public Pix2Pix pipeline
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(model_id="hustvl/pix2pix-edges2cats"):
    with st.spinner("Loading Pix2Pix model (CPU, may take ~1-2 min)..."):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch.float32
        )
        pipe.to(device)
        pipe.enable_attention_slicing()  # reduces memory usage on CPU
    return pipe

try:
    pipe = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎨 CPU Pix2Pix Sketch-to-Image")
st.write("Upload a sketch and generate a photorealistic image using a CPU-friendly Pix2Pix model.")

# Sketch upload
uploaded = st.file_uploader("Upload sketch (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    sketch = Image.open(uploaded).convert("RGB")
    st.image(sketch, caption="Input Sketch", use_column_width=True)
else:
    sketch = None
    st.info("Please upload a sketch to continue.")

# Prompt and generation parameters
prompt = st.text_area("Prompt", value="A realistic cat with detailed fur")
negative_prompt = st.text_area("Negative prompt", value="blurry, low quality", height=60)
num_steps = st.slider("Steps", 10, 25, 20)  # reduced for CPU
guidance = st.slider("Guidance scale", 1.0, 20.0, 7.5)
seed_val = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)

# Generate image
if st.button("Generate Image") and sketch:
    try:
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

        # Download button
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Image", data=buf, file_name="pix2pix_result.png", mime="image/png")

    except Exception as e:
        st.error(f"Failed to generate image: {e}")
