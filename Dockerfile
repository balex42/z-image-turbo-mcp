FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install system dependencies if needed (e.g. for opencv or others)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Default environment variables
ENV TORCH_DTYPE=bfloat16
ENV ENABLE_CPU_OFFLOAD=true

# Expose the port (FastMCP SSE default is usually 8000 or configurable)
EXPOSE 8000

# Run the server
# We use python server.py which invokes mcp.run(transport="sse")
CMD ["python", "server.py"]
