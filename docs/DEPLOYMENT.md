# MultiMix AI — Cloud Deployment & Containerization Guide

This document provides operational instructions for local deployment, containerization (Docker), model provisioning, and cloud deployment targets for MultiMix AI.

---

## 1. Current Deployment Status

| Deployment Environment | Current Status | Notes |
| :--- | :--- | :--- |
| **Local CPU / GPU (Development)** | **Complete & Verified** | Full text and voice pipelines operational locally on CPU via Streamlit. |
| **Docker Container Ready** | **Configured & Ready** | Dockerfile, `.dockerignore`, and `scripts/start.sh` configured for Linux container runtime. |
| **Cloud Production Hosting** | **Deployment Ready (Pending Cloud Host Selection)** | Containerized architecture ready for deployment to GPU Cloud (RunPod, Vast.ai, HF Spaces, AWS/GCP). |

> **IMPORTANT:** MultiMix AI is currently an offline-capable, locally validated application. No live cloud deployment exists yet. Model weights are intentionally excluded from Git and Docker images to keep the repository lightweight and modular.

---

## 2. Local Execution

To run MultiMix AI directly on your local workstation:

```bash
# 1. Activate your virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Verify model provisioning
python scripts/download_models.py

# 3. Launch Streamlit
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 3. Model Provisioning (`scripts/download_models.py`)

MultiMix AI requires 5 model categories to run complete speech and text pipelines:
1. **Qwen 2.5 1.5B Instruct:** Large language model for multilingual conversational reasoning (`models/qwen2.5-1.5b-instruct`).
2. **faster-whisper large-v3-turbo:** Speech-to-text transcription engine (`models/whisper-turbo`).
3. **IndicLID:** Language identification fastText model (`models/indiclid/indiclid-model.bin`).
4. **IndicF5:** Non-autoregressive flow-matching speech synthesis model (`models/indicf5/model.safetensors`, vocab, and reference prompt).
5. **Vocos:** Neural audio vocoder for IndicF5 audio decoding (`models/indicf5/models--charactr--vocos-mel-24khz`).

### Provisioning Command:
```bash
python scripts/download_models.py
```

- **Safe Execution:** The script creates required directories, verifies existing assets, and skips redownloading valid files.
- **Repeatable:** Can be run multiple times safely without overwriting or corrupting valid local assets.
- **Explicit Execution:** The script is never invoked automatically during standard development tests or unit runs.

---

## 4. Docker Containerization

MultiMix AI includes a standardized `Dockerfile`, `.dockerignore`, and container startup script (`scripts/start.sh`).

### 4.1 Linux System Dependencies
The container builds upon `python:3.12-slim` and installs essential native media and compilation packages:
- `ffmpeg`: Audio transcoding, decoding, and slicing for Whisper and Librosa.
- `libsndfile1`: Low-level audio I/O backend for SoundFile.
- `build-essential` & `git`: C/C++ compilation tools required for native extensions (such as FastText).
- `curl`: Network diagnostics and health checking.

### 4.2 Building the Docker Image
```bash
# Build the image from project root (excluding model weights via .dockerignore)
docker build -t multimix-ai:latest .
```

### 4.3 Running the Docker Container
Because models are excluded from the Docker image, the recommended workflow mounts a persistent model directory from the host:

```bash
# Run container with mounted models directory (CPU Mode)
docker run -d \
  --name multimix-ai-app \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  multimix-ai:latest
```

### 4.4 Startup Script & Dynamic Model Provisioning
The container entrypoint executes `scripts/start.sh`, which:
1. Verifies that `assets/audio/generated`, `assets/audio/uploads`, and `models` exist.
2. Checks the `PROVISION_MODELS` environment variable. If set to `1` or `true`, it automatically executes `scripts/download_models.py` before starting the server.
3. Binds Streamlit to `0.0.0.0` and listens on `${PORT:-8501}`.

```bash
# Run with automatic on-first-start model download:
docker run -d \
  --name multimix-ai-auto \
  -p 8501:8501 \
  -e PROVISION_MODELS=1 \
  -v multimix_models_volume:/app/models \
  multimix-ai:latest
```

---

## 5. Hardware & GPU Recommendations

| Metric | CPU Execution (Testing / Demo) | GPU Execution (Recommended Production) |
| :--- | :--- | :--- |
| **Primary Use** | Unit testing, CI/CD, functional demonstration | Interactive user experience, low latency |
| **ASR Latency (Whisper)** | ~30 – 90 seconds | ~1 – 3 seconds |
| **LLM Reasoning (Qwen)** | ~2 – 5 seconds | < 500 ms |
| **TTS Synthesis (IndicF5)** | ~90 – 180 seconds | ~2 – 4 seconds |
| **Recommended Hardware** | 8+ vCPU cores, 16+ GB RAM | 1x NVIDIA A10G / L4 / RTX 4090 (24 GB VRAM) |

> **NOTE:** CPU execution is fully supported and verified for development, testing, and functional demonstrations. GPU acceleration is strongly recommended for interactive voice workloads due to the computational demands of the 32-step ODE flow-matching solver in IndicF5.

---

## 6. Cloud Deployment Targets

### 6.1 RunPod / Vast.ai (Dedicated GPU Instances)
For optimal interactive performance and dedicated GPU access:
1. Deploy a container instance using template: `nvidia/cuda:12.1.1-devel-ubuntu22.04` or the custom MultiMix Docker image.
2. Attach a **Persistent Network Volume** (at least 30 GB) mounted to `/app/models`.
3. Set environment variables:
   - `PORT=8501`
   - `PROVISION_MODELS=1` (for initial startup only)
4. Expose HTTP port 8501 to external traffic.

### 6.2 Hugging Face Spaces (Docker Space)
1. Create a new Space selecting the **Docker** SDK.
2. Configure hardware to **Nvidia A10G Large** (recommended) or **ZeroGPU**.
3. In Space Settings, allocate persistent storage under `/data` or mount models directory.
4. Set secret/variable `PORT=7860` (Hugging Face Spaces default web port). Streamlit and `scripts/start.sh` automatically detect `$PORT`.

### 6.3 Kubernetes / AWS ECS / Google Cloud Run
- **Port:** Uses the standard Cloud Run / PaaS `$PORT` environment variable.
- **Health Check:** HTTP GET on `/_stcore/health`.
- **Persistent Volume Claim:** Bind persistent storage for `/app/models` to prevent downloading weights on every container cold start.

---

## 7. Storage, Audio Artifacts & Concurrency

- **UUID Audio Filenames:** All generated audio files use UUIDs (`response_<uuid>.wav`) to prevent race conditions and audio overwrites when multiple users interact simultaneously.
- **Audio Cleanup:** Uploaded audio (`assets/audio/uploads/`) and generated audio (`assets/audio/generated/`) should be cleared periodically via a scheduled cron job or container restart.
- **No In-Tree Weights:** Git repository size remains < 10 MB, strictly complying with cloud deployment best practices.
