"""
Monica TTS Server - Chatterbox Multilingual
Serveur de synthèse vocale pour Bourly Poker Tour.
Endpoint POST /blabla : reçoit du texte, renvoie du WAV.
"""

import io
import os
import logging
import threading
import time
from contextlib import asynccontextmanager

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REFERENCE_AUDIO = os.getenv("REFERENCE_AUDIO", "26-monica--interview.wav")
REFERENCE_DURATION_S = int(os.getenv("REFERENCE_DURATION_S", "120"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "500"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("monica-tts")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
ref_audio_path: str | None = None
device: str = "cpu"
inference_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_device() -> str:
    """Detect best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        dev = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"CUDA detected: {gpu_name} ({vram:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        logger.info("Apple Metal (MPS) detected")
    else:
        dev = "cpu"
        logger.info("No GPU detected, using CPU (inference will be slow)")
    return dev


def prepare_reference_audio(src_path: str, duration_s: int) -> str:
    """Crop reference audio to `duration_s` seconds and save as temp file."""
    logger.info(f"Loading reference audio: {src_path}")
    waveform, sr = ta.load(src_path)

    max_samples = sr * duration_s
    if waveform.shape[1] > max_samples:
        logger.info(
            f"Cropping reference from {waveform.shape[1]/sr:.1f}s to {duration_s}s"
        )
        waveform = waveform[:, :max_samples]
    else:
        logger.info(f"Reference audio is {waveform.shape[1]/sr:.1f}s (no crop needed)")

    out_path = "/tmp/monica_reference_10s.wav"
    ta.save(out_path, waveform, sr)
    logger.info(f"Reference audio ready: {out_path} ({duration_s}s, {sr}Hz)")
    return out_path


def load_model(dev: str):
    """Load ChatterboxMultilingualTTS model.

    The chatterbox library's from_pretrained() calls torch.load() without
    map_location, so checkpoints saved on CUDA crash on MPS/CPU.
    We monkey-patch torch.load to force map_location='cpu' during loading,
    then the library moves tensors to the target device itself.
    """
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    logger.info(f"Loading ChatterboxMultilingualTTS on device={dev} ...")
    t0 = time.time()

    if dev != "cuda":
        # Monkey-patch torch.load to handle CUDA-saved checkpoints on CPU/MPS
        _original_load = torch.load

        def _patched_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = "cpu"
            return _original_load(*args, **kwargs)

        torch.load = _patched_load
        try:
            mdl = ChatterboxMultilingualTTS.from_pretrained(device=dev)
        finally:
            torch.load = _original_load
    else:
        mdl = ChatterboxMultilingualTTS.from_pretrained(device=dev)

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.1f}s")
    return mdl


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, ref_audio_path, device

    # 1. Detect device
    device = detect_device()

    # 2. Prepare reference audio
    ref_audio_path = prepare_reference_audio(REFERENCE_AUDIO, REFERENCE_DURATION_S)

    # 3. Load model
    model = load_model(device)

    # Warm-up: run a short inference to pre-compile graphs
    logger.info("Warm-up inference ...")
    with inference_lock:
        _ = model.generate(
            "Test.",
            audio_prompt_path=ref_audio_path,
            language_id="fr",
        )
    logger.info("Server ready!")

    yield

    # Shutdown
    logger.info("Shutting down ...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Monica TTS - Bourly Poker Tour",
    description="Text-to-Speech server using Chatterbox Multilingual (French)",
    version="1.0.0",
    lifespan=lifespan,
)


class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text to synthesize in French",
        examples=["Pablo vient d'éliminer Rusmir avec une paire d'as!"],
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": device,
        "model_loaded": model is not None,
    }


@app.post("/blabla")
async def generate_speech(request: TTSRequest):
    """
    Generate WAV speech from text.
    Compatible with PokerTourIRL tts.service.ts:
      POST { "text": "..." } → binary WAV
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars, max {MAX_TEXT_LENGTH})",
        )

    logger.info(f"Generating TTS for: {text[:80]}{'...' if len(text) > 80 else ''}")
    t0 = time.time()

    try:
        with inference_lock:
            wav = model.generate(
                text,
                audio_prompt_path=ref_audio_path,
                language_id="fr",
            )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed")

    # Convert tensor to WAV bytes
    buffer = io.BytesIO()
    ta.save(buffer, wav, model.sr, format="wav")
    buffer.seek(0)
    wav_bytes = buffer.read()

    elapsed = time.time() - t0
    logger.info(f"Generated {len(wav_bytes)} bytes in {elapsed:.2f}s")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="monica.wav"',
            "X-Generation-Time": f"{elapsed:.2f}s",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        workers=1,  # Single worker: model is not fork-safe
    )
