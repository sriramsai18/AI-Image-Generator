import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

import os

# ─── MODEL LOAD ───────────────────────────────────────────────────────────────
# Detect if we should use mock mode (default to true on CPU to prevent out-of-memory or slow load)
MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1" or not torch.cuda.is_available()

if MOCK_MODE:
    print("Running in MOCK MODE (MOCK_MODE=1 or no GPU available) 🎨")
    pipe = None
else:
    print("Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,          # removes NSFW filter delay
        requires_safety_checker=False
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # speed optimisation for CPU
    if not torch.cuda.is_available():
        pipe.enable_attention_slicing()

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
            # Generate a beautiful placeholder gradient/accent image
            from PIL import ImageDraw
            img = Image.new("RGB", (int(width), int(height)), color="#0d1117")
            draw = ImageDraw.Draw(img)

            # Draw a subtle high-contrast border
            draw.rectangle([8, 8, int(width)-8, int(height)-8], outline="#e63946", width=2)

            # Draw aesthetic futuristic grid lines
            for i in range(4):
                x = 8 + (int(width) - 16) * (i + 1) // 5
                draw.line([x, 8, x, int(height)-8], fill="#1f1618", width=1)
                y = 8 + (int(height) - 16) * (i + 1) // 5
                draw.line([8, y, int(width)-8, y], fill="#1f1618", width=1)

            # Simple decorative corner brackets
            corner_len = 20
            draw.line([8, 8, 8 + corner_len, 8], fill="#39ff14", width=3)
            draw.line([8, 8, 8, 8 + corner_len], fill="#39ff14", width=3)
            draw.line([int(width)-8, 8, int(width)-8 - corner_len, 8], fill="#39ff14", width=3)
            draw.line([int(width)-8, 8, int(width)-8, 8 + corner_len], fill="#39ff14", width=3)
            draw.line([8, int(height)-8, 8 + corner_len, int(height)-8], fill="#39ff14", width=3)
            draw.line([8, int(height)-8, 8, int(height)-8 - corner_len], fill="#39ff14", width=3)
            draw.line([int(width)-8, int(height)-8, int(width)-8 - corner_len, int(height)-8], fill="#39ff14", width=3)
            draw.line([int(width)-8, int(height)-8, int(width)-8, int(height)-8 - corner_len], fill="#39ff14", width=3)

            # Print details using standard font
            draw.text((24, 24), "STABLE DIFFUSION v1.5", fill="#e63946")
            draw.text((24, 54), f"PROMPT: {prompt[:40]}...", fill="#d4dde8")
            draw.text((24, 84), f"NEGATIVE: {negative_prompt[:40]}...", fill="#8a9ab0")
            draw.text((24, 114), f"SEED: {seed if seed != -1 else 'random'}", fill="#39ff14")
            draw.text((24, 144), f"STEPS: {steps} | CFG: {guidance}", fill="#d4dde8")
            draw.text((24, int(height) - 40), "[ MOCKED GENERATION ]", fill="#39ff14")
            image = img
        else:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(width),
                height=int(height),
                generator=generator,
            )
            image = result.images[0]

        elapsed = round(time.time() - start, 1)
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

/* Keyboard Navigation & Accessibility Focus Indicators */
button.primary:focus-visible {
    outline: 3px solid #39ff14 !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 20px rgba(57, 255, 20, 0.8) !important;
}
button.secondary:focus-visible {
    outline: 3px solid #e63946 !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 20px rgba(230, 57, 70, 0.8) !important;
}
textarea:focus-visible, input:focus-visible, select:focus-visible {
    outline: 2px solid #e63946 !important;
    outline-offset: 1px !important;
    border-color: #e63946 !important;
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

/* Reset / Secondary button */
button.secondary {
    background: #0f1318 !important;
    border: 1px solid rgba(230,57,70,0.3) !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #8a9ab0 !important;
    transition: all 0.3s !important;
}
button.secondary:hover {
    background: #151a21 !important;
    border-color: #e63946 !important;
    color: #fff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 0 15px rgba(230,57,70,0.2) !important;
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

            with gr.Row():
                generate_btn = gr.Button("▶ GENERATE IMAGE", variant="primary", size="lg", scale=2)
                reset_btn = gr.Button("🔄 RESET", variant="secondary", size="lg", scale=1)

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

    def reset_all_fields():
        return (
            "",                                               # prompt
            "blurry, ugly, distorted, low quality, watermark", # negative_prompt
            25,                                               # steps
            7.5,                                              # guidance
            512,                                              # width
            512,                                              # height
            -1,                                               # seed
            None,                                             # output_image
            ""                                                # info_text
        )

    # ── BIND ────────────────────────────────────────────────────────────────
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
        outputs=[output_image, info_text],
    )
    reset_btn.click(
        fn=reset_all_fields,
        inputs=[],
        outputs=[prompt, negative_prompt, steps, guidance, width, height, seed, output_image, info_text],
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
    demo.launch(server_name="0.0.0.0", server_port=3000)
