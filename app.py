import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import time
import os

# ─── MODEL LOAD ───────────────────────────────────────────────────────────────
MOCK_MODE = os.environ.get("MOCK_MODE") == "1"

if MOCK_MODE:
    print("MOCK_MODE is enabled! Setting up a lightweight offline mockup...")
    class MockPipelineResult:
        def __init__(self, images):
            self.images = images

    class MockModule:
        def to(self, *args, **kwargs):
            return self

    class MockPipeline:
        def __init__(self):
            self.unet = MockModule()
            self.vae = MockModule()

        def to(self, device):
            return self

        def enable_attention_slicing(self):
            pass

        def __call__(self, prompt, negative_prompt=None, num_inference_steps=25, guidance_scale=7.5, width=512, height=512, generator=None):
            # Create a structured PIL canvas with a high-contrast mockup layout
            image = Image.new("RGB", (width, height), color="#0f1318")
            draw = ImageDraw.Draw(image)
            # Draw aesthetic details
            draw.rectangle([10, 10, width-10, height-10], outline="#e63946", width=2)
            draw.line([10, 50, width-10, 50], fill="#e63946", width=1)
            draw.text((20, 20), "OFFLINE MOCK MODE GENERATION", fill="#e63946")

            # Display prompt info
            text_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt
            draw.text((20, 70), f"Prompt: {text_prompt}", fill="#d4dde8")
            draw.text((20, 100), f"Steps: {num_inference_steps}  CFG: {guidance_scale}", fill="#8a9ab0")
            draw.text((20, 130), f"Resolution: {width}x{height}", fill="#8a9ab0")

            # Draw a colorful abstract center decoration
            center_x, center_y = width // 2, height // 2 + 30
            draw.ellipse([center_x - 50, center_y - 50, center_x + 50, center_y + 50], outline="#39ff14", width=2)
            draw.text((center_x - 35, center_y - 5), "AI PIL Canvas", fill="#39ff14")

            return MockPipelineResult([image])

    pipe = MockPipeline()
    print("Mock model loaded ✅")
else:
    print("Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,          # removes NSFW filter delay
        requires_safety_checker=False
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Apply memory layout optimization (channels_last) to UNet and VAE layers
# This delivers higher convolution performance on PyTorch (CPU and GPU)
try:
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    print("Optimized memory formatting (channels_last) applied ✅")
except Exception as e:
    print(f"Skipped channels_last memory format optimization: {e}")

# Conditionally configure attention slicing based on physical system RAM.
# Attention slicing (enable_attention_slicing) introduces severe CPU overhead (up to ~230% slowdown)
# and must be avoided on CPU unless RAM is extremely constrained (< 4GB).
if not torch.cuda.is_available() and not MOCK_MODE:
    try:
        total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        total_ram_gb = total_ram / (1024 ** 3)
    except (AttributeError, ValueError):
        # Fallback if os.sysconf is not supported
        total_ram_gb = 8.0

    if total_ram_gb < 4.0:
        print(f"System RAM is extremely constrained ({total_ram_gb:.2f} GB). Enabling attention slicing to conserve memory...")
        pipe.enable_attention_slicing()
    else:
        print(f"System RAM is sufficient ({total_ram_gb:.2f} GB). Skipping attention slicing to avoid CPU overhead (up to 230% speedup) ✅")

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
        # Use torch.inference_mode() for accelerated inference without autograd overhead
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
