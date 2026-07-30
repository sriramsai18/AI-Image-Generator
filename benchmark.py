import time

import torch
from diffusers import StableDiffusionPipeline


def main():
    print("Initializing benchmark with tiny-stable-diffusion-torch on CPU...")
    # Using a tiny model to make the benchmark incredibly fast and low on memory usage
    model_id = "hf-internal-testing/tiny-stable-diffusion-torch"

    # 1. Baseline Run
    print("\n--- Running Baseline (No Optimizations) ---")
    pipe_baseline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe_baseline = pipe_baseline.to("cpu")

    # Enable attention slicing for baseline to measure its overhead
    pipe_baseline.enable_attention_slicing()

    # Warmup
    _ = pipe_baseline("warmup prompt", num_inference_steps=2)

    # Run
    start = time.time()
    for _ in range(3):
        _ = pipe_baseline("test prompt", num_inference_steps=5, width=128, height=128)
    baseline_time = time.time() - start
    print(f"Baseline Time: {baseline_time:.4f}s")

    # 2. Optimized Run
    print("\n--- Running Optimized (Channels Last, No Attention Slicing, Inference Mode) ---")
    pipe_opt = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe_opt = pipe_opt.to("cpu")

    # Optimizations
    if hasattr(pipe_opt, "unet"):
        pipe_opt.unet.to(memory_format=torch.channels_last)
    if hasattr(pipe_opt, "vae"):
        pipe_opt.vae.to(memory_format=torch.channels_last)

    # Warmup
    with torch.inference_mode():
        _ = pipe_opt("warmup prompt", num_inference_steps=2)

    # Run
    start = time.time()
    with torch.inference_mode():
        for _ in range(3):
            _ = pipe_opt("test prompt", num_inference_steps=5, width=128, height=128)
    opt_time = time.time() - start
    print(f"Optimized Time: {opt_time:.4f}s")

    # Speedup calculation
    speedup = (baseline_time - opt_time) / baseline_time * 100
    print(f"\nOptimization speedup: {speedup:.2f}% faster!")

if __name__ == "__main__":
    main()
