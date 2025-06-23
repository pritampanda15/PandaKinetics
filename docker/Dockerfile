# =============================================================================
# Docker/Dockerfile - GPU-enabled container
# =============================================================================

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements*.txt ./
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir cupy-cuda12x

# Copy source code
COPY . .

# Install PandaKinetics
RUN pip install --no-cache-dir -e ".[gpu]"

# Create non-root user
RUN useradd -m -u 1000 pandauser && \
    chown -R pandauser:pandauser /app
USER pandauser

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["python", "-c", "import pandakinetics; print('PandaKinetics ready!')"]
