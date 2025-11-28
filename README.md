# Z-Image MCP Server

This is a Model Context Protocol (MCP) server for the Z-Image-Turbo model, running in Docker.

## Prerequisites

- Docker
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit

## Running with Docker Compose

1. Build and start the server:
   ```bash
   docker compose up --build
   ```

2. The server will be available at `http://localhost:8000/sse`.

## Configuration

You can modify `compose.yaml` to change environment variables:

- `TORCH_DTYPE`: Set to `float16` for older GPUs (like RTX 6000) or `bfloat16` for newer ones (Ampere+).
- `ENABLE_CPU_OFFLOAD`: Set to `true` to save VRAM, or `false` to keep model in VRAM for faster inference.

## Usage

This server exposes a `generate_image` tool that accepts:
- `prompt`: Text description
- `height`: Image height (default 1024)
- `width`: Image width (default 1024)
- `steps`: Inference steps (default 9)

It returns the image as a base64-encoded PNG.
