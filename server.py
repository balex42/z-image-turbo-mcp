from fastmcp import FastMCP, Context
from fastmcp.utilities.types import Image
import torch
from diffusers import ZImagePipeline
import io
import os
import threading
import asyncio

# Initialize FastMCP
mcp = FastMCP("z-image-server")

# Global pipeline variable
pipe = None
gpu_lock = threading.Lock()

async def load_model(ctx: Context):
    global pipe
    if pipe is not None:
        return

    print("Loading Z-Image-Turbo model...")
    await ctx.info("Loading Z-Image-Turbo model...")

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Memory optimizations
    # Use environment variable to control offloading if needed, default to True
    if os.getenv("ENABLE_CPU_OFFLOAD", "true").lower() == "true":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    print("Model loaded successfully.")
    await ctx.info("Model loaded successfully.")

# Default configuration from environment variables
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "9"))
env_seed = os.getenv("DEFAULT_SEED")
DEFAULT_SEED = int(env_seed) if env_seed else None
DEFAULT_NUM_IMAGES = int(os.getenv("DEFAULT_NUM_IMAGES", "1"))

def run_inference(prompt, num_outputs, generators, callback):
    with gpu_lock:
        return pipe(
            prompt=prompt,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=0.0,
            num_images_per_prompt=num_outputs,
            generator=generators,
            callback_on_step_end=callback
        ).images

@mcp.tool()
async def generate_image(prompt: str, ctx: Context) -> list[Image | str]:
    """Generate an image from a rich natural‑language prompt using Z-Image-Turbo.

    This tool works best when the calling LLM provides a **detailed** prompt,
    not just a single word. The prompt should describe, in one or more
    sentences, things like:

    - main subject(s) and what they are doing
    - environment / background (indoor, outdoor, landscape, city, etc.)
    - style (photo, illustration, 3D render, pixel art, anime, watercolor, etc.)
    - composition and camera details (close‑up, full‑body, wide shot, angle)
    - lighting and mood (soft light, dramatic, neon, sunny, cozy, etc.)
    - any important colors, level of realism, or extra details to emphasize

    The more specific and concrete the prompt, the better the results.

    Args:
        prompt: A rich, descriptive text prompt containing subject, style,
                environment, and other key visual details.
    """
    await load_model(ctx)

    # Determine base seed (allow reproducible outputs when DEFAULT_SEED is set)
    if DEFAULT_SEED is None:
        base_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        base_seed = int(DEFAULT_SEED)

    # Prepare per-image generators so each image is different but reproducible
    num_outputs = DEFAULT_NUM_IMAGES
    seeds = [base_seed + i for i in range(num_outputs)]
    generators = [torch.Generator("cuda").manual_seed(s) for s in seeds]

    print(f"Generating {num_outputs} image(s) for prompt: {prompt} with base seed: {base_seed}")
    # Use lock to ensure only one generation happens at a time
    await ctx.info(f"Waiting for GPU lock to generate image...")
    
    loop = asyncio.get_running_loop()
    
    def callback(pipe, step_index, timestep, callback_kwargs):
        asyncio.run_coroutine_threadsafe(
            ctx.report_progress(progress=step_index + 1, total=DEFAULT_STEPS),
            loop
        )
        return callback_kwargs

    await ctx.info(f"Generating image")
    await ctx.report_progress(progress=0, total=DEFAULT_STEPS)
    
    images = await asyncio.to_thread(
        run_inference, 
        prompt, 
        num_outputs, 
        generators, 
        callback
    )

    # Convert each PIL image to raw PNG bytes and wrap in the Image type
    output_images = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        output_images.append(Image(data=buffered.getvalue(), format="png"))

    # Return list of images and a short message
    return output_images + [f"Generated {num_outputs} image(s) for prompt: {prompt}"]

if __name__ == "__main__":
    # Run with HTTP transport (Modern "Streamable")
    mcp.run(transport="http", host="0.0.0.0", port=8000)
