# MultiMix AI — Deployment Guide & Infrastructure Roadmap

This document outlines the current deployment status of MultiMix AI and details the target architecture for future production cloud deployment.

---

## 1. Current Deployment Status

| Deployment Environment | Current Status | Notes |
| :--- | :--- | :--- |
| **Local CPU (MVP)** | **Complete & Verified** | Full text and voice pipelines operational locally on CPU via Streamlit. |
| **Cloud GPU Server** | **Not Deployed Yet** | Architecture designed; pending target cloud host selection and provisioning. |
| **Public GitHub Repository** | **Not Finalized Yet** | Local codebase prepared, verified, and documented; ready for remote publication. |

MultiMix AI is currently an **offline-capable, locally runnable Minimum Viable Product (MVP)**. It has not yet been deployed to public cloud services (such as AWS, GCP, or Hugging Face Spaces).

---

## 2. Local MVP Architecture

Currently, all subsystems run in a single Python environment on the host machine:

```
[User Browser]
      │
      ▼ (HTTP / WebSocket)
[Streamlit Server (app.py)]
      │
      ├──> faster-whisper large-v3-turbo (Local CPU INT8)
      ├──> Language Detection & Segmentation (Local Rule Engine + fastText)
      ├──> Normalization & Semantic Grounding (Local Rule Engine)
      ├──> Qwen 2.5 1.5B Instruct (Local CPU Float32)
      └──> IndicF5 TTS & Vocos Vocoder (Local CPU Float32)
```

### Operational Characteristics of Local CPU Deployment
- **Advantages:** Complete data privacy, zero external cloud costs, offline operation, zero API key dependencies.
- **Trade-offs:** High latency on CPU (ASR ~50-100s, TTS ~2-3 min per request).

---

## 3. Future Cloud GPU Production Architecture

To transition MultiMix AI from a local MVP to a production service with sub-second response times, the recommended target deployment architecture decouples the front-end user interface from GPU inference microservices.

```mermaid
flowchart TD
    Client[Web & Mobile Clients] --> LB[Load Balancer / Cloudflare]
    LB --> FE[Streamlit Web Frontend / FastAPI Gateway]
    
    subgraph GPU Inference Cluster [GPU Server: NVIDIA A10G / A100 / L4]
        FE -->|Audio Stream| ASR_SVC[faster-whisper Service (CUDA FP16)]
        ASR_SVC -->|Transcription Text| NLP_SVC[Multilingual Segmentation & Semantic Engine]
        NLP_SVC -->|Grounded Prompt| LLM_SVC[Qwen 2.5 1.5B (vLLM / Hugging Face TGI)]
        LLM_SVC -->|Response Text| TTS_SVC[IndicF5 + Vocos (CUDA FP16)]
        TTS_SVC -->|Audio Output (WAV)| FE
    end

    subgraph Storage & Cache
        FE <--> S3[Object Storage: S3 / GCS (Audio Artifacts)]
        LLM_SVC <--> REDIS[(Redis Cache / Session State)]
    end
```

### Recommended Infrastructure Specifications
- **GPU Accelerator:** 1x NVIDIA A10G (24 GB VRAM) or NVIDIA L4 (24 GB VRAM).
- **Host System:** 4-8 vCPU, 32 GB System RAM.
- **OS / Container:** Ubuntu 22.04 LTS with NVIDIA Container Toolkit (Docker).
- **Serving Stack:**
  - *LLM Serving:* `vLLM` or Hugging Face Text Generation Inference (TGI) for high-throughput batching.
  - *ASR Serving:* CTranslate2 with CUDA execution provider.
  - *TTS Serving:* PyTorch with CUDA acceleration (reducing ODE solver latency to < 2 seconds).

---

## 4. Containerization Roadmap (Docker)

A future production deployment should containerize the service using a multi-stage Dockerfile:

```dockerfile
# Concept Dockerfile for GPU Deployment
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 5. Security & Network Considerations for Cloud Deployment

When deploying MultiMix AI to cloud environments:
1. **Model Storage:** Host models on cloud object storage (S3 / GCS) or pull directly from Hugging Face Hub during container provisioning rather than embedding weights inside container images.
2. **Audio File Retention:** Implement an automatic TTL (Time to Live) on uploaded audio files and generated speech outputs to protect user data privacy.
3. **Transport Encryption:** Terminate TLS/SSL at the load balancer or reverse proxy (Nginx / Cloudflare).
