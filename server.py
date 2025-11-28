from fastmcp import FastMCP
from mcp.types import ImageContent
import torch
from diffusers import ZImagePipeline
import base64
import io
import os
import threading

# Initialize FastMCP
mcp = FastMCP("z-image-server")

# Global pipeline variable
pipe = None
gpu_lock = threading.Lock()

def load_model():
    global pipe
    if pipe is not None:
        return

    print("Loading Z-Image-Turbo model...")
    dtype_str = os.getenv("TORCH_DTYPE", "bfloat16")
    
    # Handle dtype selection
    if dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
        
    print(f"Using dtype: {dtype}")

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # Memory optimizations
    # Use environment variable to control offloading if needed, default to True
    if os.getenv("ENABLE_CPU_OFFLOAD", "true").lower() == "true":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print("Model loaded successfully.")

# Default configuration from environment variables
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "9"))
env_seed = os.getenv("DEFAULT_SEED")
DEFAULT_SEED = int(env_seed) if env_seed else None

@mcp.tool()
def generate_image(prompt: str) -> ImageContent:
    """
    Generate an image based on a text prompt using Z-Image-Turbo.
    
    Args:
        prompt: The text description of the image to generate.
        height: Height of the image (default from env DEFAULT_HEIGHT or 1024).
        width: Width of the image (default from env DEFAULT_WIDTH or 1024).
        steps: Number of inference steps (default from env DEFAULT_STEPS or 9).
        seed: Optional seed for reproducibility. Default from env DEFAULT_SEED or random.
    """
    load_model()
    
    if DEFAULT_SEED is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    print(f"Generating image for prompt: {prompt} with seed: {seed}")
    
    # Use lock to ensure only one generation happens at a time
    with gpu_lock:
        image = pipe(
            prompt=prompt,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return ImageContent(
        type="image",
        data=img_str,
        mimeType="image/png"
    )

if __name__ == "__main__":
    # Run with HTTP transport (Modern "Streamable")
    mcp.run(transport="http", host="0.0.0.0", port=8000)
