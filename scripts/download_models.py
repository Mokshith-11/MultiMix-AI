#!/usr/bin/env python3
"""
MultiMix AI — Model Provisioning Script

Downloads and verifies all runtime model dependencies for cloud and local deployment.
Safe for repeated execution: skips existing complete models unless --force is specified.

Usage:
    python scripts/download_models.py [--force]
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    QWEN_MODEL_PATH,
    INDICF5_MODEL_DIR,
    WHISPER_TURBO_MODEL_DIR,
    INDICLID_MODEL_PATH,
)


def log_status(model_name: str, status: str, detail: str = ""):
    prefix = f"[{status}]"
    print(f"  {prefix:<7} {model_name}" + (f" — {detail}" if detail else ""))


def provision_qwen(force: bool = False) -> bool:
    print("\n--- 1. Qwen 2.5 1.5B Instruct ---")
    QWEN_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    config_file = QWEN_MODEL_PATH / "config.json"
    weights_safetensors = QWEN_MODEL_PATH / "model.safetensors"
    weights_index = QWEN_MODEL_PATH / "model.safetensors.index.json"

    if not force and config_file.exists() and (weights_safetensors.exists() or weights_index.exists()):
        log_status("Qwen 2.5 1.5B", "EXISTS", str(QWEN_MODEL_PATH))
        return True

    print("  Downloading Qwen 2.5 1.5B Instruct from Hugging Face...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct",
            local_dir=str(QWEN_MODEL_PATH),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        log_status("Qwen 2.5 1.5B", "OK", "Download complete")
        return True
    except Exception as e:
        log_status("Qwen 2.5 1.5B", "FAILED", str(e))
        return False


def provision_whisper_turbo(force: bool = False) -> bool:
    print("\n--- 2. faster-whisper large-v3-turbo ---")
    WHISPER_TURBO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    has_model = (
        (WHISPER_TURBO_MODEL_DIR / "model.bin").exists()
        or (WHISPER_TURBO_MODEL_DIR / "blobs").exists()
        or bool(list(WHISPER_TURBO_MODEL_DIR.glob("models--*faster-whisper-large-v3-turbo")))
    )

    if not force and has_model:
        log_status("faster-whisper", "EXISTS", str(WHISPER_TURBO_MODEL_DIR))
        return True

    print("  Downloading faster-whisper large-v3-turbo...")
    try:
        from faster_whisper import download_model
        download_model(
            "large-v3-turbo",
            output_dir=str(WHISPER_TURBO_MODEL_DIR),
        )
        log_status("faster-whisper", "OK", "Download complete")
        return True
    except Exception as e:
        log_status("faster-whisper", "FAILED", str(e))
        return False


def provision_indiclid(force: bool = False) -> bool:
    print("\n--- 3. IndicLID (AI4Bharat Romanized LID) ---")
    INDICLID_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force and INDICLID_MODEL_PATH.exists() and INDICLID_MODEL_PATH.stat().st_size > 1000000:
        log_status("IndicLID", "EXISTS", str(INDICLID_MODEL_PATH))
        return True

    print("  Downloading IndicLID fastText binary from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="ai4bharat/IndicLID",
            filename="indiclid-ft-topk/model_baseline_roman.bin",
            local_dir=str(INDICLID_MODEL_PATH.parent),
        )
        # Move if downloaded into subfolder
        downloaded = INDICLID_MODEL_PATH.parent / "indiclid-ft-topk" / "model_baseline_roman.bin"
        if downloaded.exists() and not INDICLID_MODEL_PATH.exists():
            downloaded.replace(INDICLID_MODEL_PATH)

        log_status("IndicLID", "OK", "Download complete")
        return True
    except Exception as e:
        log_status("IndicLID", "FAILED", str(e))
        return False


def provision_indicf5(force: bool = False) -> bool:
    print("\n--- 4. IndicF5 TTS (AI4Bharat) ---")
    INDICF5_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_file = INDICF5_MODEL_DIR / "model.safetensors"
    vocab_file = INDICF5_MODEL_DIR / "checkpoints" / "vocab.txt"

    if not force and model_file.exists() and vocab_file.exists():
        log_status("IndicF5", "EXISTS", str(INDICF5_MODEL_DIR))
        return True

    print("  Downloading IndicF5 checkpoint and vocabulary...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="ai4bharat/IndicF5",
            local_dir=str(INDICF5_MODEL_DIR),
        )
        log_status("IndicF5", "OK", "Download complete")
        return True
    except Exception as e:
        log_status("IndicF5", "FAILED", str(e))
        return False


def provision_vocos(force: bool = False) -> bool:
    print("\n--- 5. Vocos 24kHz Neural Vocoder ---")
    try:
        from vocos import Vocos
        print("  Verifying/downloading Vocos 24kHz pretrained weights...")
        vocos_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        log_status("Vocos Vocoder", "OK", "Vocos 24kHz ready")
        return True
    except Exception as e:
        log_status("Vocos Vocoder", "FAILED", str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="MultiMix AI Model Provisioning Utility")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    print("=" * 70)
    print("MultiMix AI — Model Provisioning")
    print(f"Target models directory: {MODELS_DIR}")
    print(f"Force re-download: {args.force}")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        provision_qwen(force=args.force),
        provision_whisper_turbo(force=args.force),
        provision_indiclid(force=args.force),
        provision_indicf5(force=args.force),
        provision_vocos(force=args.force),
    ]

    print("\n" + "=" * 70)
    if all(results):
        print("SUCCESS: All deployment model assets are provisioned and verified.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("ERROR: One or more models failed to provision. Check details above.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
