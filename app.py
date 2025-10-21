import os
import io
import random
from typing import Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import cv2

# diffusers
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# ---------------------------
# Utility functions
# ---------------------------

def ensure_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # optional: support MPS (mac)
    if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_sketch(pil_img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Convert the input sketch to a fixed-size, single-channel 'scribble' suitable
    for the ControlNet scribble model. Steps:
      - convert to grayscale
      - invert if necessary (controlnet scribble expects dark lines on white background)
      - threshold / normalize
      - center-crop or pad to keep aspect ratio, then resize
    """
    # convert to gray
    img = pil_img.convert("L")

    # auto-detect if background is dark -> invert
    # compute median brightness
    median = np.median(np.array(img))
    if median < 127:
        img = ImageOps.invert(img)

    # apply a slight blur then adaptive threshold with OpenCV for clearer lines
    arr = np.array(img)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    # adaptive threshold
    th = cv2.adaptiveThreshold(
        arr,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    img = Image.fromarray(th)

    # keep aspect ratio: pad to square then resize
    w, h = img.size
    new_w, new_h = target_size
    # make square by padding
    if w != h:
        max_side = max(w, h)
        padded = Image.new("L", (max_side, max_side), color=255)  # white background
        paste_x = (max_side - w) // 2
        paste_y = (max_side - h) // 2
        padded.paste(img, (paste_x, paste_y))
        img = padded

    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


# ---------------------------
# Pipeline loading (cached)
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline(
    sd_model_id: str,
    controlnet_model_id: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Loads ControlNet model and Stable Diffusion pipeline from Hugging Face.
    Caching avoids re-downloading on every run.
    """
    # load controlnet
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype)
    # load sd pipeline with controlnet plugged in
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id,
        controlnet=controlnet,
        safety_checker=None,  # set to None or implement your own safety checker if desired
        torch_dtype=dtype,
    )
    # use UniPC scheduler (fast + good quality) — replace scheduler config if needed
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # VAE may be loaded with the pipeline automatically
    pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None

    pipe.to(device)
    if device.type == "cuda" and dtype == torch.float16:
        # warm-up (optional)
        pass
    return pipe


# ---------------------------
# Inference
# ---------------------------

def generate_from_sketch(
    pipe: StableDiffusionControlNetPipeline,
    sketch_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    strength: float,
    width: int,
    height: int,
    device: torch.device,
):
    """
    Run the pipeline and return a PIL.Image
    Parameters:
      - strength: how much the controlnet scribble influences the output (0..1)
    """
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    # diffusers expects RGB sketch for some controlnet variants — convert accordingly
    # but the scribble controlnet accepts single-channel images as well
    if sketch_pil.mode != "RGB":
        sketch_rgb = sketch_pil.convert("RGB")
    else:
        sketch_rgb = sketch_pil

    # run pipeline
    with torch.autocast(device.type if device.type != "cpu" else "cpu"):
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() != "" else None,
            image=sketch_rgb,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    images = out.images
    return images[0] if images else None


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Sketch → Image (Streamlit + ControlNet)", layout="centered")
st.title("🎨 Sketch → Image — Streamlit + Stable Diffusion + ControlNet")

st.markdown(
    """
Upload a hand-drawn or digital sketch and convert it into a photorealistic image.
- **Model**: Stable Diffusion + ControlNet (scribble / line art)
- **Notes**: GPU recommended. Provide your Hugging Face token via `HUGGINGFACEHUB_API_TOKEN` env var or in the field below.
"""
)

# Sidebar: Hugging Face token and model choices
st.sidebar.header("Configuration")
hf_token_from_env = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
hf_token = st.sidebar.text_input(
    "Hugging Face token (optional, recommended for model downloads)",
    value=hf_token_from_env,
    type="password",
)
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

device = ensure_device()
st.sidebar.write(f"Device: **{device.type}**")
use_half = st.sidebar.checkbox("Use float16 (faster on CUDA GPUs)", value=True if device.type == "cuda" else False)
dtype = torch.float16 if (use_half and device.type == "cuda") else torch.float32

# model selection (defaults are proven combos)
st.sidebar.subheader("Models")
sd_model_id = st.sidebar.text_input(
    "Stable Diffusion model id",
    value="runwayml/stable-diffusion-v1-5",
    help="Hugging Face model repo (e.g. runwayml/stable-diffusion-v1-5)"
)
controlnet_model_id = st.sidebar.text_input(
    "ControlNet model id (scribble/line art)",
    value="lllyasviel/sd-controlnet-scribble",
    help="ControlNet model repo for sketches (scribble). Alternatives: lllyasviel/sd-controlnet-canny"
)

# load pipeline (cached)
with st.spinner("Loading models (may take a while the first time)..."):
    try:
        pipe = load_pipeline(sd_model_id, controlnet_model_id, device=device, dtype=dtype)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# Main UI: upload sketch
st.subheader("1) Upload your sketch")
uploaded = st.file_uploader("Upload an image (PNG/JPG). Prefer black or dark strokes on white background.", type=["png","jpg","jpeg"])
use_example = st.button("Use example sketch")
if use_example and not uploaded:
    # simple built-in example (drawn strokes)
    example_img = Image.new("L", (512,512), color=255)
    draw = Image.fromarray(np.array(example_img))
    uploaded = io.BytesIO()
    example_img.save(uploaded, format="PNG")
    uploaded.seek(0)

if uploaded:
    raw = Image.open(uploaded)
    st.markdown("**Original upload**")
    st.image(raw, use_column_width=True)
else:
    st.info("Upload a sketch to begin, or click 'Use example sketch'.")

# Preprocess / resize options
st.subheader("2) Preprocessing & generation settings")
col1, col2 = st.columns([1,1])
with col1:
    out_res = st.selectbox("Output resolution", options=["512x512","768x768"], index=0)
    width, height = map(int, out_res.split("x"))
with col2:
    auto_resize = st.checkbox("Auto-preprocess sketch (recommended)", value=True)

if uploaded:
    preview_size = (width, height)
    preprocessed = preprocess_sketch(raw, preview_size) if auto_resize else raw.convert("L").resize(preview_size)
    st.markdown("**Preprocessed sketch (fed to ControlNet)**")
    st.image(preprocessed, width=320)

# Prompt controls
st.subheader("3) Prompt & generation parameters")
prompt = st.text_area("Prompt (describe the scene/result you want)", value="A photorealistic portrait of a young woman, highly detailed, studio lighting, 85mm", height=100)
negative_prompt = st.text_area("Negative prompt (what to avoid)", value="low quality, deformed, ugly, text, watermark", height=60)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    steps = st.slider("Steps", min_value=10, max_value=100, value=28, step=1)
with col2:
    guidance = st.slider("Guidance scale (CFG)", min_value=1.0, max_value=20.0, value=7.5)
with col3:
    seed_val = st.number_input("Seed (0 for random)", min_value=0, value=0, step=1)

# Generate button
st.subheader("4) Generate")
if st.button("Generate image from sketch"):
    if not uploaded:
        st.warning("Please upload a sketch first.")
    else:
        seed = None if seed_val == 0 else int(seed_val)
        with st.spinner("Generating… this can take 10–90 seconds depending on hardware and steps"):
            try:
                # preprocessed is available
                result_img = generate_from_sketch(
                    pipe=pipe,
                    sketch_pil=preprocessed,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed if seed is not None else random.randint(1, 2**31 - 1),
                    strength=1.0,
                    width=width,
                    height=height,
                    device=device,
                )
                if result_img is None:
                    st.error("Generation returned no image.")
                else:
                    st.success("Done!")
                    st.image(result_img, caption="Generated image", use_column_width=True)
                    buf = io.BytesIO()
                    result_img.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button("Download result (PNG)", data=buf, file_name="sketch2image.png", mime="image/png")
            except Exception as e:
                st.error(f"Generation failed: {e}")
