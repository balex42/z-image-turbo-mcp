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

class Config:
    HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
    WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
    STEPS = int(os.getenv("DEFAULT_STEPS", "9"))
    SEED = int(os.getenv("DEFAULT_SEED")) if os.getenv("DEFAULT_SEED") else None
    ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "true").lower() == "true"

class ZImageEngine:
    def __init__(self):
        self.pipe = None
        self.lock = threading.Lock()

    async def load_model(self, ctx: Context):
        if self.pipe is not None:
            return

        await ctx.info("Loading Z-Image-Turbo model...")

        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        if Config.ENABLE_CPU_OFFLOAD:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

        await ctx.info("Model loaded successfully.")

    def run_inference(self, prompt, generator, callback, start_callback=None):
        with self.lock:
            if start_callback:
                start_callback()
            return self.pipe(
                prompt=prompt,
                height=Config.HEIGHT,
                width=Config.WIDTH,
                num_inference_steps=Config.STEPS,
                guidance_scale=0.0,
                num_images_per_prompt=1,
                generator=generator,
                callback_on_step_end=callback
            ).images[0]

engine = ZImageEngine()

@mcp.tool()
async def generate_image(prompt: str, ctx: Context) -> list[Image | str]:
    """Generate an image from a rich natural-language prompt using Z-Image-Turbo.

    Language: Prompts must be provided in English.

    This tool works best when the calling LLM provides a detailed prompt,
    not just a single word. The prompt should describe, in one or more
    sentences, things like:

    - main subject(s) and what they are doing
    - environment / background (indoor, outdoor, landscape, city, etc.)
    - style (photo, illustration, 3D render, pixel art, anime, watercolor, etc.)
    - composition and camera details (close-up, full-body, wide shot, angle)
    - lighting and mood (soft light, dramatic, neon, sunny, cozy, etc.)
    - any important colors, level of realism, or extra details to emphasize
    - enclose all wanted text content within the image in double quotes (" ")

    The more specific and concrete the prompt, the better the results.

    Args:
        prompt: A rich, descriptive text prompt (in English) containing subject, style,
                environment, and other key visual details.
    """
    await engine.load_model(ctx)

    # Determine base seed (allow reproducible outputs when Config.SEED is set)
    if Config.SEED is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        seed = int(Config.SEED)

    # Prepare random generator
    generator = torch.Generator("cuda").manual_seed(seed)

    # Use lock to ensure only one generation happens at a time
    await ctx.info(f"Waiting for GPU lock to generate image...")
    
    loop = asyncio.get_running_loop()
    
    def callback(pipe, step_index, timestep, callback_kwargs):
        step = step_index + 1
        asyncio.run_coroutine_threadsafe(
            ctx.report_progress(progress=step, total=Config.STEPS),
            loop
        )
        asyncio.run_coroutine_threadsafe(
            ctx.info(f"Generating image: step {step}/{Config.STEPS}"),
            loop
        )
        return callback_kwargs

    def start_callback():
        asyncio.run_coroutine_threadsafe(ctx.info(f"Starting image generation..."), loop)
        asyncio.run_coroutine_threadsafe(ctx.report_progress(progress=0, total=Config.STEPS), loop)
    
    image = await asyncio.to_thread(
        engine.run_inference, 
        prompt, 
        generator, 
        callback,
        start_callback
    )

    # Convert PIL image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    # Return image and success message
    return [Image(data=buffered.getvalue(), format="png"), f"Successfully generated image for prompt: {prompt}"]

if __name__ == "__main__":
    # Run with HTTP transport (Modern "Streamable")
    mcp.run(transport="http", host="0.0.0.0", port=8000)
