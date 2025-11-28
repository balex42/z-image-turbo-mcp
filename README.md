# Z-Image Turbo – MCP Server

An HTTP Model Context Protocol (MCP) server that hosts the Diffusers Z-Image-Turbo pipeline behind a simple MCP tool. Runs in Docker with NVIDIA GPU acceleration.

The server exposes one tool:
- `generate_image(prompt: str) -> list[Image | str]`

It returns a list of PNG images (MCP Image type, raw PNG bytes) and a short message. Image size, steps, and number of outputs are controlled via environment variables.

## Prerequisites

- Docker
- NVIDIA GPU with recent drivers
- NVIDIA Container Toolkit (to pass the GPU into containers)

## Quick start (Docker Compose)

Build and start the server:

```bash
docker compose up --build -d
```

The MCP HTTP endpoint will be available at:

- Base URL: `http://localhost:8000`
- MCP path: `http://localhost:8000/mcp`

Note: This is an MCP server, not a REST API. Use an MCP client (see below) or the included Python client to call tools.

To view logs or stop:

```bash
docker compose logs -f
docker compose down
```

## Configuration

All configuration is handled via environment variables (see `compose.yaml`). Defaults are applied in `server.py`.

- `ENABLE_CPU_OFFLOAD` (default: `true`)
   - `true`: Offload layers to CPU to reduce VRAM usage (slower, more memory transfer)
   - `false`: Keep model fully on GPU for maximum speed
- `DEFAULT_HEIGHT` (default: `1024`)
- `DEFAULT_WIDTH` (default: `1024`)
- `DEFAULT_STEPS` (default: `9`)
- `DEFAULT_SEED` (optional; if set, outputs are reproducible)
- `DEFAULT_NUM_IMAGES` (default: `4`) – number of images generated per prompt

Other relevant Compose settings:

- The Hugging Face cache is persisted via a bind mount: `${HOME}/.cache/huggingface:/root/.cache/huggingface`
- GPU is reserved via `deploy.resources.reservations.devices` with the NVIDIA driver.

## Using the included Python client

You can test the server with the provided client in `my_client.py`. It connects to `http://localhost:8000/mcp` using the FastMCP Python client and invokes the `generate_image` tool.

Run it locally (outside the container) after the server is up:

```bash
python3 my_client.py
```

If you want to save the returned images, you can adapt the client like so:

```python
import asyncio
from fastmcp import Client
from fastmcp.utilities.types import Image

client = Client("http://localhost:8000/mcp")

async def call_and_save(prompt: str):
      async with client:
            results = await client.call_tool("generate_image", {"prompt": prompt})
            # Results is a list of Image objects (PNG bytes) and a trailing message string
            img_idx = 0
            for item in results:
                  if isinstance(item, Image):
                        with open(f"output_{img_idx}.png", "wb") as f:
                              f.write(item.data)
                        img_idx += 1
            print(results[-1])  # e.g., "Generated N image(s) for prompt: ..."

asyncio.run(call_and_save("A futuristic cityscape at sunset"))
```

## MCP client integration

Any MCP-aware client that supports HTTP transport can connect to `http://localhost:8000/mcp` and call the `generate_image` tool with a single `prompt` string. Image size, steps, and batch size are controlled via the environment variables listed above (not per-call parameters).

## What’s running under the hood?

- `server.py` starts a FastMCP HTTP server on `0.0.0.0:8000` and lazily loads the Diffusers `ZImagePipeline` (`Tongyi-MAI/Z-Image-Turbo`).
- CPU offload can be toggled with `ENABLE_CPU_OFFLOAD`.
- A GPU lock ensures a single generation runs at a time to avoid VRAM spikes.

## Troubleshooting

- Container can’t see the GPU
   - Ensure NVIDIA drivers and Container Toolkit are installed.
   - On some hosts you may need to restart Docker after installing the toolkit.
- CUDA / driver mismatch errors
   - The image uses CUDA 12.8 runtimes. Make sure your host drivers support CUDA 12.
- Out-of-memory (OOM) on GPU
   - Set `ENABLE_CPU_OFFLOAD=true`.
   - Reduce `DEFAULT_NUM_IMAGES` and/or lower `DEFAULT_HEIGHT`/`DEFAULT_WIDTH`.
- Slow first request
   - The first prompt loads weights into memory; subsequent calls are faster.
- Hugging Face rate limits or auth
   - If you need private models, mount a volume with your `huggingface` cache and login token as needed.

## Development notes

- Python deps are pinned in `requirements.txt` and installed in the Docker image.
- The service listens on port 8000 (mapped in Compose). Adjust if needed.
