import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import time
import os
import random

# Determine if we should bypass loading the model for offline/mock development
MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"

# ─── MODEL LOAD ───────────────────────────────────────────────────────────────
if MOCK_MODE:
    print("MOCK_MODE is active. Bypassing real model loading. ✅")
    pipe = None
else:
    print("Loading Stable Diffusion v1.5...")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,          # removes NSFW filter delay
        requires_safety_checker=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Apply 'channels_last' memory format optimization to UNet and VAE layers.
    # This optimizes memory access patterns, significantly improving inference speed on compatible GPUs and CPUs.
    try:
        if hasattr(pipe, "unet") and pipe.unet is not None:
            pipe.unet.to(memory_format=torch.channels_last)
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(memory_format=torch.channels_last)
        print("Optimized memory layout (channels_last) for UNet and VAE layers. ⚡")
    except Exception as e:
        print(f"Could not apply channels_last optimization: {e}")

    # speed optimisation for CPU
    if not torch.cuda.is_available():
        # Attention slicing (enable_attention_slicing) introduces severe CPU overhead (up to ~230% slowdown)
        # and should be avoided on CPU unless RAM is extremely constrained (<4GB).
        # We check the system memory using standard python libraries to conditionally enable it only if total RAM is less than 4GB.
        try:
            import os
            # Retrieve total physical memory bytes using standard os.sysconf
            page_size = os.sysconf('SC_PAGE_SIZE')
            phys_pages = os.sysconf('SC_PHYS_PAGES')
            total_ram_gb = (page_size * phys_pages) / (1024 ** 3)
        except Exception:
            # Fallback if sysconf is not supported on the host platform
            total_ram_gb = 8.0  # Assume reasonable default to avoid attention slicing

        if total_ram_gb < 4.0:
            print(f"System RAM ({total_ram_gb:.1f}GB) is extremely constrained (<4GB). Enabling attention slicing.")
            pipe.enable_attention_slicing()
        else:
            print(f"System RAM ({total_ram_gb:.1f}GB) is sufficient (>=4GB). Avoiding attention slicing to prevent severe CPU overhead. ⚡")

    print("Model loaded ✅")

# ─── GENERATION FUNCTION ──────────────────────────────────────────────────────
def generate_image(prompt, negative_prompt, steps, guidance, width, height, seed):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt first!"

    generator = None
    if seed != -1:
        generator = torch.Generator().manual_seed(int(seed))

    try:
        start = time.time()

        if MOCK_MODE:
            # Generate a structured, lightweight offline mockup PIL canvas
            w, h = int(width), int(height)
            # Deterministic color generation based on prompt/seed
            seed_val = int(seed) if seed != -1 else hash(prompt) % 10000
            random.seed(seed_val)
            r = random.randint(20, 80)
            g = random.randint(20, 80)
            b = random.randint(20, 80)

            image = Image.new("RGB", (w, h), color=(r, g, b))
            draw = ImageDraw.Draw(image)

            # Draw cross-grid lines for structured look
            grid_size = 64
            for x in range(0, w, grid_size):
                draw.line([(x, 0), (x, h)], fill=(min(255, r+20), min(255, g+20), min(255, b+20)), width=1)
            for y in range(0, h, grid_size):
                draw.line([(0, y), (w, y)], fill=(min(255, r+20), min(255, g+20), min(255, b+20)), width=1)

            # Draw inner border
            draw.rectangle([(10, 10), (w-10, h-10)], outline=(min(255, r+40), min(255, g+40), min(255, b+40)), width=2)

            # Draw mockup details text
            text_lines = [
                "⚡ [MOCK MODE] GENERATED IMAGE",
                f"Prompt: {prompt[:35]}...",
                f"Size: {w}x{h} | Seed: {seed if seed != -1 else 'random'}",
                f"Steps: {steps} | Guidance: {guidance}"
            ]
            y_offset = h // 4
            for line in text_lines:
                draw.text((w // 8, y_offset), line, fill=(255, 255, 255))
                y_offset += 30

            elapsed = 0.1
        else:
            # Wrap inference with torch.inference_mode() to skip tracking gradients,
            # reducing overhead and memory consumption for maximum performance.
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt.strip() else None,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    width=int(width),
                    height=int(height),
                    generator=generator,
                )
            elapsed = round(time.time() - start, 1)
            image = result.images[0]

        info  = f"✅ Generated in {elapsed}s  |  Steps: {steps}  |  CFG: {guidance}  |  Seed: {seed if seed != -1 else 'random'}"
        return image, info

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ─── EXAMPLE PROMPTS ──────────────────────────────────────────────────────────
examples = [
    ["a lone tree in a golden wheat field at sunset, dramatic lighting, 4k", "blurry, ugly, distorted", 25, 7.5, 512, 512, -1],
    ["a futuristic cyberpunk city at night, neon lights, rain, cinematic", "blurry, low quality", 30, 8.0, 512, 512, -1],
    ["a majestic snow-capped mountain reflected in a crystal clear lake, hyperrealistic", "cartoon, painting", 25, 7.5, 512, 512, -1],
    ["portrait of an astronaut on Mars, dramatic lighting, photorealistic", "ugly, deformed", 28, 7.5, 512, 512, -1],
    ["a cozy cafe interior with warm lighting and coffee cups, aesthetic", "blurry, dark", 25, 7.0, 512, 512, -1],
]

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

body, .gradio-container {
    background: #080b0f !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 28px 0 10px;
    border-bottom: 1px solid rgba(230,57,70,0.2);
    margin-bottom: 24px;
}
.app-title {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    color: #fff;
    text-shadow: 0 0 30px rgba(230,57,70,0.4);
    letter-spacing: 2px;
}
.app-title span { color: #e63946; }
.app-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #8a9ab0;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* Inputs */
.gradio-container label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #e63946 !important;
}
textarea, input[type="text"], input[type="number"] {
    background: #0f1318 !important;
    border: 1px solid rgba(230,57,70,0.25) !important;
    border-radius: 6px !important;
    color: #d4dde8 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
textarea:focus, input:focus {
    border-color: #e63946 !important;
    box-shadow: 0 0 12px rgba(230,57,70,0.2) !important;
}

/* Generate button */
button.primary {
    background: linear-gradient(135deg, #e63946, #c1121f) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: white !important;
    box-shadow: 0 0 20px rgba(230,57,70,0.35) !important;
    transition: all 0.3s !important;
}
button.primary:hover {
    box-shadow: 0 0 35px rgba(230,57,70,0.6) !important;
    transform: translateY(-2px) !important;
}

/* Output image panel */
.output-image {
    border: 1px solid rgba(230,57,70,0.2) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Sliders */
input[type="range"] { accent-color: #e63946 !important; }

/* Info textbox */
.info-box textarea {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #39ff14 !important;
    background: #0a0f0a !important;
    border-color: rgba(57,255,20,0.2) !important;
}

/* Accordion / panels */
.gr-box, .gr-panel {
    background: #0f1318 !important;
    border: 1px solid rgba(230,57,70,0.12) !important;
    border-radius: 10px !important;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 16px 0 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #8a9ab0;
    letter-spacing: 2px;
    border-top: 1px solid rgba(230,57,70,0.1);
    margin-top: 20px;
}
.app-footer a { color: #e63946; text-decoration: none; }
.app-footer a:hover { text-decoration: underline; }
"""

# ─── GRADIO UI ────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="Text2Image — Sriram") as demo:

    gr.HTML("""
    <div class="app-header">
        <div class="app-title">TEXT <span>2</span> IMAGE</div>
        <div class="app-sub">// Stable Diffusion v1.5 &nbsp;·&nbsp; RunwayML &nbsp;·&nbsp; Built by Sriram Sai</div>
    </div>
    """)

    with gr.Row():

        # ── LEFT: Controls ──────────────────────────────────────────────────
        with gr.Column(scale=1):

            prompt = gr.Textbox(
                label="PROMPT",
                placeholder="describe what you want to generate...",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="NEGATIVE PROMPT  (what to avoid)",
                placeholder="blurry, ugly, low quality, distorted...",
                lines=2,
                value="blurry, ugly, distorted, low quality, watermark"
            )

            with gr.Accordion("⚙️ ADVANCED SETTINGS", open=False):
                with gr.Row():
                    steps    = gr.Slider(10, 50, value=25, step=1,   label="INFERENCE STEPS")
                    guidance = gr.Slider(1,  20, value=7.5, step=0.5, label="GUIDANCE SCALE (CFG)")
                with gr.Row():
                    width  = gr.Slider(256, 768, value=512, step=64, label="WIDTH (px)")
                    height = gr.Slider(256, 768, value=512, step=64, label="HEIGHT (px)")
                seed = gr.Number(value=-1, label="SEED  (-1 = random)")

            generate_btn = gr.Button("▶ GENERATE IMAGE", variant="primary", size="lg")

            gr.Examples(
                examples=examples,
                inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
                label="💡 EXAMPLE PROMPTS — click to load",
                examples_per_page=5,
            )

        # ── RIGHT: Output ───────────────────────────────────────────────────
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="GENERATED IMAGE",
                type="pil",
                elem_classes=["output-image"],
                height=520,
            )
            info_text = gr.Textbox(
                label="STATUS",
                interactive=False,
                elem_classes=["info-box"],
                lines=1,
            )

    # ── BIND ────────────────────────────────────────────────────────────────
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
        outputs=[output_image, info_text],
    )
    prompt.submit(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
        outputs=[output_image, info_text],
    )

    gr.HTML("""
    <div class="app-footer">
        Built by <a href="https://github.com/sriramsai18" target="_blank">Sriram Sai Laggisetti</a>
        &nbsp;·&nbsp;
        Model: <a href="https://huggingface.co/runwayml/stable-diffusion-v1-5" target="_blank">runwayml/stable-diffusion-v1-5</a>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/sriram-sai-laggisetti/" target="_blank">LinkedIn</a>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
