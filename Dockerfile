# =============================================================================
# Monica TTS Server - Chatterbox Multilingual
# Multi-device: NVIDIA CUDA (primary), Apple MPS (fallback), CPU (last resort)
# =============================================================================

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install PyTorch with CUDA 12.4 support
# The wheels bundle CUDA runtime libs, no need for nvidia/cuda base image.
# The NVIDIA driver + nvidia-container-toolkit must be on the HOST.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install chatterbox-tts (will reuse already-installed torch)
RUN pip install --no-cache-dir chatterbox-tts

# Install server dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
COPY server.py .
COPY 26-monica--interview.wav .

# Pre-download model weights at build time (optional, ~2 GB).
# Comment out the next line to download at first startup instead.
RUN python -c "\
from chatterbox.mtl_tts import ChatterboxMultilingualTTS; \
ChatterboxMultilingualTTS.from_pretrained(device='cpu')" \
    || echo 'Model pre-download failed, will retry at startup'

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run server
CMD ["python", "server.py"]
