from pathlib import Path
from typing import Optional

import sys
import soundfile as sf


from src.config import PROJECT_ROOT, INDICF5_MODEL_DIR, GENERATED_AUDIO_DIR

# ============================================================
# MultiMix AI - IndicF5 TTS Engine
# ============================================================

INDICF5_ROOT = INDICF5_MODEL_DIR

MODEL_PATH = INDICF5_ROOT / "model.safetensors"
VOCAB_PATH = INDICF5_ROOT / "checkpoints" / "vocab.txt"
REFERENCE_AUDIO = INDICF5_ROOT / "prompts" / "PAN_F_HAPPY_00001.wav"

REFERENCE_TEXT = "Namaste, aaj aap kaise hain?"

OUTPUT_DIR = GENERATED_AUDIO_DIR

MODEL_CONFIG = {
    "dim": 1024,
    "depth": 22,
    "heads": 16,
    "ff_mult": 2,
    "text_dim": 512,
    "conv_layers": 4,
}


_model = None
_vocoder = None


def _load_engine():
    """
    Load IndicF5 model and vocoder once.
    Subsequent calls reuse the loaded objects.
    """

    global _model
    global _vocoder

    if _model is not None and _vocoder is not None:
        return _model, _vocoder

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"IndicF5 model not found: {MODEL_PATH}"
        )

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"IndicF5 vocabulary not found: {VOCAB_PATH}"
        )

    if not REFERENCE_AUDIO.exists():
        raise FileNotFoundError(
            f"IndicF5 reference audio not found: {REFERENCE_AUDIO}"
        )

    # Allow Python to import the downloaded IndicF5 package.
    if str(INDICF5_ROOT) not in sys.path:
        sys.path.insert(0, str(INDICF5_ROOT))

    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import (
        load_vocoder,
        load_model,
    )

    print("Loading IndicF5 vocoder...")

    _vocoder = load_vocoder()

    print("Loading IndicF5 model...")

    _model = load_model(
        DiT,
        MODEL_CONFIG,
        str(MODEL_PATH),
        vocab_file=str(VOCAB_PATH),
    )

    print("IndicF5 TTS engine loaded.")

    return _model, _vocoder


def synthesize_speech(
    text: str,
    output_path: Optional[str] = None,
    speed: float = 1.0,
) -> str:
    """
    Convert text into speech using IndicF5.

    Parameters
    ----------
    text:
        Text that should be spoken.

    output_path:
        Optional output WAV path.

    speed:
        Speech speed. 1.0 = normal.

    Returns
    -------
    str
        Path of generated WAV file.
    """

    text = text.strip()

    if not text:
        raise ValueError(
            "Cannot generate speech from empty text."
        )

    model, vocoder = _load_engine()

    from f5_tts.infer.utils_infer import infer_process

    if output_path is None:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            OUTPUT_DIR / "multimix_response.wav"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Generating IndicF5 speech for: {text}"
    )

    final_wave, sample_rate, _ = infer_process(
        str(REFERENCE_AUDIO),
        REFERENCE_TEXT,
        text,
        model,
        vocoder,
        cross_fade_duration=0.15,
        speed=speed,
    )

    sf.write(
        str(output_path),
        final_wave,
        sample_rate,
    )

    print(
        f"Speech generated: {output_path}"
    )

    return str(output_path)