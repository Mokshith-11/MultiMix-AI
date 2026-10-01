from pathlib import Path
import base64
import os
import tempfile
import uuid

import modal


# ============================================================
# MultiMix AI - Modal GPU Backend
# ============================================================

ROOT = Path(__file__).resolve().parent
DEPLOY_SRC = ROOT / "deployment" / "huggingface_backend" / "src"


# ============================================================
# Hugging Face Secret
# ============================================================

hf_secret = modal.Secret.from_name(
    "multmixai",
    environment_name="main",
    required_keys=["HF_TOKEN"],
)


# ============================================================
# CUDA Container Image
# ============================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "ffmpeg",
        "git",
    )
    .pip_install(
        # CUDA-enabled PyTorch
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",

        # Hugging Face / Transformers
        "transformers==4.49.0",
        "accelerate>=1.10.0",
        "safetensors>=0.4.0",
        "sentencepiece>=0.2.0",
        "huggingface_hub>=0.34.0",

        # Whisper
        "faster-whisper>=1.1.0",
        "ctranslate2>=4.5.0",

        # API / audio
        "fastapi>=0.115.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",

        # IndicF5-compatible NumPy
        "numpy==1.26.4",

        # IndicF5 dependencies
        "pypinyin",
        "ema-pytorch",
        "vocos",
        "torchdiffeq",
        "einops",
        "x-transformers",
        "jieba",
        "cached_path",
        "datasets",
        "hydra-core",
        "tomli",
        "tqdm",
        "wandb",
        "transformers-stream-generator",

        # Language detection
        "fasttext-wheel==0.9.2",

        # Audio / ML utilities
        "librosa>=0.10.0",

        # Official AI4Bharat IndicF5 package
        "git+https://github.com/AI4Bharat/IndicF5.git",

        # PyTorch CUDA wheel index
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(
        DEPLOY_SRC,
        "/root/src",
        copy=True,
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
        ],
    )
)


# ============================================================
# Modal Application
# ============================================================

app = modal.App(
    "multimix-ai-backend",
    secrets=[hf_secret],
)


# ============================================================
# Helper: Audio -> Base64
# ============================================================

def _audio_to_base64(audio_path: str) -> str:
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Generated audio file not found: {audio_path}"
        )

    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


# ============================================================
# Helper: API-Safe Result
# ============================================================

def _json_safe_result(result: dict) -> dict:
    output = dict(result)

    audio_path = output.pop(
        "response_audio_path",
        None,
    )

    if audio_path:
        output["audio_base64"] = _audio_to_base64(
            audio_path
        )

        output["audio_mime"] = "audio/wav"

        output["audio_filename"] = (
            f"multimix_response_{uuid.uuid4().hex}.wav"
        )

    else:
        output["audio_base64"] = None
        output["audio_mime"] = None
        output["audio_filename"] = None

    return output


# ============================================================
# Health Endpoint
# ============================================================

@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=30,
    scaledown_window=60,
)
@modal.fastapi_endpoint(
    method="GET",
)
def health():

    hf_available = bool(
        os.getenv("HF_TOKEN")
    )

    return {
        "status": "ok",
        "service": "MultiMix AI GPU Backend",
        "backend": "Modal",
        "gpu": "T4",
        "hf_token_configured": hf_available,
        "pipeline": [
            "ASR",
            "Language Detection",
            "Normalization",
            "Semantic Interpretation",
            "Qwen",
            "Translation",
            "IndicF5",
        ],
    }


# ============================================================
# Main Inference Endpoint
# ============================================================

@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=32768,
    timeout=900,
    startup_timeout=1200,
    max_containers=1,
    scaledown_window=120,
)
@modal.fastapi_endpoint(
    method="POST",
)
def process(payload: dict):

    # Import only inside the remote container.
    from src.voice.pipeline import (
        process_text,
        process_voice,
    )

    mode = str(
        payload.get("mode", "text")
    ).strip().lower()

    target_language = str(
        payload.get("target_language", "Auto")
    ).strip()

    if target_language.lower() == "auto":
        target_language = ""


    # ========================================================
    # TEXT MODE
    # ========================================================

    if mode == "text":

        text = str(
            payload.get("text", "")
        ).strip()

        if not text:
            return {
                "status": "error",
                "error": "Text input is empty.",
            }

        try:

            result = process_text(
                text=text,
                target_language=target_language,
            )

            return _json_safe_result(result)

        except Exception as error:

            return {
                "status": "error",
                "error": str(error),
                "mode": "text",
            }


    # ========================================================
    # VOICE MODE
    # ========================================================

    if mode == "voice":

        audio_base64 = payload.get(
            "audio_base64"
        )

        filename = str(
            payload.get(
                "filename",
                "input.wav",
            )
        )

        if not audio_base64:
            return {
                "status": "error",
                "error": "audio_base64 is missing.",
            }

        try:

            audio_bytes = base64.b64decode(
                audio_base64
            )

        except Exception as error:

            return {
                "status": "error",
                "error": f"Invalid base64 audio: {error}",
            }

        suffix = Path(filename).suffix

        if not suffix:
            suffix = ".wav"

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp:

                temp.write(audio_bytes)
                temp_path = temp.name

            result = process_voice(
                audio_path=temp_path,
                target_language=target_language,
            )

            return _json_safe_result(result)

        except Exception as error:

            return {
                "status": "error",
                "error": str(error),
                "mode": "voice",
            }

        finally:

            if temp_path:

                try:
                    Path(temp_path).unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass


    # ========================================================
    # INVALID MODE
    # ========================================================

    return {
        "status": "error",
        "error": (
            "Unsupported mode. "
            "Use 'text' or 'voice'."
        ),
    }


# ============================================================
# Local Entrypoint
# ============================================================

@app.local_entrypoint()
def main():

    print("=" * 60)
    print("MULTIMIX AI - MODAL GPU BACKEND")
    print("=" * 60)

    print("Backend : Modal")
    print("GPU     : T4")
    print(f"Source  : {DEPLOY_SRC}")
    print("Secret  : multmixai")
    print("Env     : main")

    print()

    if not DEPLOY_SRC.exists():
        raise FileNotFoundError(
            f"Deployment source not found: {DEPLOY_SRC}"
        )

    print("Deployment source exists")
    print("Modal configuration loaded")
    print("Hugging Face secret configured")
    print("IndicF5 package configured")
    print()
    print(
        "Development:"
    )
    print(
        "python -m modal serve modal_backend.py"
    )
    print()
    print(
        "Production:"
    )
    print(
        "python -m modal deploy modal_backend.py"
    )
