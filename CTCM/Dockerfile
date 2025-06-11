# Dockerfile for Neural Operator Continuous Time Consistency Model

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    tmux \
    htop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data checkpoints samples logs

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV WANDB_MODE=offline

# Default command
CMD ["/bin/bash"]