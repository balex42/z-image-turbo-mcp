FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install only what we really need
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-pip \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create python → python3 symlink (fixes "python: not found")
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install PyTorch with CUDA 12.8 support
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual code
COPY server.py .

# Default env vars (you can override them at runtime)
ENV ENABLE_CPU_OFFLOAD=true

EXPOSE 8000

# Use the real binary path (works everywhere)
CMD ["/usr/bin/python3", "server.py"]
