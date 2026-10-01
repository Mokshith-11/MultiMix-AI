from pathlib import Path
from typing import Optional
import os

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel

from src.config import (
    GENERATED_AUDIO_DIR,
    INDICF5_MODEL_ID,
    REFERENCE_TEXT,
)

_model = None
_reference_audio = None


def _get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not available inside the Modal container."
        )

    return token


def _load_engine():
    global _model
    global _reference_audio

    if _model is not None and _reference_audio is not None:
        return _model, _reference_audio

    print("Loading IndicF5...")

    hf_token = _get_hf_token()

    _reference_audio = hf_hub_download(
        repo_id=INDICF5_MODEL_ID,
        filename="prompts/PAN_F_HAPPY_00001.wav",
        token=hf_token,
    )

    print("IndicF5 reference audio downloaded.")

    _model = AutoModel.from_pretrained(
        INDICF5_MODEL_ID,
        token=hf_token,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        _model = _model.to("cuda")

    _model.eval()

    print("IndicF5 loaded.")

    return _model, _reference_audio


def synthesize_speech(
    text: str,
    output_path: Optional[str] = None,
    speed: float = 1.0,
) -> str:

    text = text.strip()

    if not text:
        raise ValueError(
            "Cannot synthesize empty text."
        )

    model, reference_audio = _load_engine()

    if output_path is None:
        GENERATED_AUDIO_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            GENERATED_AUDIO_DIR / "response.wav"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Generating IndicF5 speech: {text}"
    )

    audio = model(
        text,
        ref_audio_path=reference_audio,
        ref_text=REFERENCE_TEXT,
    )

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    audio = np.asarray(audio)

    if audio.dtype == np.int16:
        audio = (
            audio.astype(np.float32) / 32768.0
        )
    else:
        audio = audio.astype(np.float32)

    sf.write(
        str(output_path),
        audio,
        samplerate=24000,
    )

    print(
        f"Speech generated: {output_path}"
    )

    return str(output_path)
