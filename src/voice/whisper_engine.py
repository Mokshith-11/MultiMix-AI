"""
MultiMix AI — ASR engine using faster-whisper (large-v3-turbo).

Replaces the original OpenAI Whisper Small implementation with
CTranslate2-backed faster-whisper for dramatically better accuracy
on short, code-mixed, multilingual audio clips.

Public API (unchanged):
    load_whisper_model()  → WhisperModel
    transcribe_audio()    → dict
    get_device()          → str

Added in Task 3:
    validate_asr_quality() → dict
"""

from pathlib import Path
from typing import Dict, Optional

from faster_whisper import WhisperModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from src.config import WHISPER_TURBO_MODEL_DIR

MODEL_PATH = WHISPER_TURBO_MODEL_DIR
MODEL_SIZE = "large-v3-turbo"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Singleton model cache
_model: Optional[WhisperModel] = None


# ---------------------------------------------------------------------------
# ASR Quality Thresholds
# ---------------------------------------------------------------------------
# These thresholds are based on observed faster-whisper metrics:
#
#   Good transcription (code-mixed speech):
#     avg_logprob ≈ -0.08 to -0.22, no_speech_prob ≈ 0.0
#
#   Garbage transcription (old Whisper Small on same audio):
#     avg_logprob ≈ -1.88, no_speech_prob ≈ 0.10
#
# Thresholds are deliberately lenient to avoid rejecting legitimate
# multilingual/code-mixed speech which may have lower confidence than
# clean single-language audio.
# ---------------------------------------------------------------------------

# "good" requires ALL of these to pass
GOOD_AVG_LOGPROB_MIN = -0.7       # above this → healthy ASR
GOOD_NO_SPEECH_PROB_MAX = 0.3     # below this → speech is present
GOOD_COMPRESSION_RATIO_MAX = 2.4  # below this → not repetitive/hallucinated

# "failed" if ANY of these trigger
FAILED_AVG_LOGPROB_MIN = -1.5     # below this → almost certainly garbage
FAILED_NO_SPEECH_PROB_MAX = 0.8   # above this → model thinks no speech
FAILED_MIN_TEXT_LENGTH = 1        # empty text → failed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_whisper_model() -> WhisperModel:
    """Load the faster-whisper large-v3-turbo model (cached singleton)."""
    global _model

    if _model is not None:
        return _model

    _model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=str(MODEL_PATH),
    )

    return _model


def validate_asr_quality(transcription: Dict) -> Dict:
    """
    Classify ASR output quality using actual faster-whisper metrics.

    Returns:
        {
            "quality": "good" | "low" | "failed",
            "reasons": [str, ...],   # human-readable explanations
        }

    This does NOT invent a confidence score. It classifies quality
    based on the model's own diagnostic signals.
    """

    text = transcription.get("text", "").strip()
    avg_logprob = transcription.get("avg_logprob")
    no_speech_prob = transcription.get("no_speech_prob")
    compression_ratio = transcription.get("compression_ratio")

    reasons = []

    # ------------------------------------------------------------------
    # Check for "failed" conditions first (any single trigger → failed)
    # ------------------------------------------------------------------

    if len(text) < FAILED_MIN_TEXT_LENGTH:
        return {
            "quality": "failed",
            "reasons": ["Empty or near-empty transcription"],
        }

    if avg_logprob is not None and avg_logprob < FAILED_AVG_LOGPROB_MIN:
        reasons.append(
            f"Very low avg_logprob ({avg_logprob:.3f} < {FAILED_AVG_LOGPROB_MIN})"
        )
        return {"quality": "failed", "reasons": reasons}

    if no_speech_prob is not None and no_speech_prob > FAILED_NO_SPEECH_PROB_MAX:
        reasons.append(
            f"High no_speech_prob ({no_speech_prob:.3f} > {FAILED_NO_SPEECH_PROB_MAX})"
        )
        return {"quality": "failed", "reasons": reasons}

    # ------------------------------------------------------------------
    # Check for "good" (all criteria must pass)
    # ------------------------------------------------------------------

    is_good = True

    if avg_logprob is not None and avg_logprob < GOOD_AVG_LOGPROB_MIN:
        reasons.append(
            f"avg_logprob below good threshold "
            f"({avg_logprob:.3f} < {GOOD_AVG_LOGPROB_MIN})"
        )
        is_good = False

    if no_speech_prob is not None and no_speech_prob > GOOD_NO_SPEECH_PROB_MAX:
        reasons.append(
            f"Elevated no_speech_prob "
            f"({no_speech_prob:.3f} > {GOOD_NO_SPEECH_PROB_MAX})"
        )
        is_good = False

    if compression_ratio is not None and compression_ratio > GOOD_COMPRESSION_RATIO_MAX:
        reasons.append(
            f"High compression_ratio "
            f"({compression_ratio:.3f} > {GOOD_COMPRESSION_RATIO_MAX})"
        )
        is_good = False

    if is_good:
        return {"quality": "good", "reasons": []}

    # ------------------------------------------------------------------
    # Everything else → "low" (usable but uncertain)
    # ------------------------------------------------------------------

    return {"quality": "low", "reasons": reasons}


def transcribe_audio(audio_path: str) -> Dict:
    """
    Transcribe speech with faster-whisper.

    The first pass strongly requests Latin/Romanized output so
    downstream multilingual detection can operate on Romanized
    Telugu, Tamil, Hindi, Bengali and English text.
    """

    model = _load_model()

    audio = _load_audio(audio_path)

    romanization_prompt = (
        "Romanized transcription only. "
        "Use Latin alphabet. "
        "Do not translate. "
        "Do not use Devanagari, Telugu, Tamil, Bengali or other Indic scripts. "
        "Preserve the original Telugu, Tamil, Hindi, Bengali and English "
        "words as spoken. "
        "Example: Nenu today college ki vellanu but my friend "
        "Tamil-la pesitu irundhan."
    )

    segments, info = model.transcribe(
        audio,
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=romanization_prompt,
    )

    segments = list(segments)

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    # Detect non-Latin output.
    has_indic_script = any(
        "\u0900" <= ch <= "\u097F" or
        "\u0C00" <= ch <= "\u0C7F" or
        "\u0B80" <= ch <= "\u0BFF" or
        "\u0980" <= ch <= "\u09FF"
        for ch in text
    )

    # Retry once with an even stronger Latin-only prompt.
    if has_indic_script:
        retry_prompt = (
            "WRITE ONLY IN LATIN LETTERS. "
            "ROMANIZED TRANSCRIPTION ONLY. "
            "DO NOT OUTPUT DEVANAGARI. "
            "DO NOT OUTPUT TELUGU SCRIPT. "
            "DO NOT OUTPUT TAMIL SCRIPT. "
            "DO NOT OUTPUT BENGALI SCRIPT. "
            "DO NOT TRANSLATE. "
            "Example: Nenu today college ki vellanu but my friend "
            "Tamil-la pesitu irundhan."
        )

        retry_segments, retry_info = model.transcribe(
            audio,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=retry_prompt,
        )

        retry_segments = list(retry_segments)

        retry_text = " ".join(
            segment.text.strip()
            for segment in retry_segments
        ).strip()

        retry_has_indic = any(
            "\u0900" <= ch <= "\u097F" or
            "\u0C00" <= ch <= "\u0C7F" or
            "\u0B80" <= ch <= "\u0BFF" or
            "\u0980" <= ch <= "\u09FF"
            for ch in retry_text
        )

        if retry_text and not retry_has_indic:
            text = retry_text
            info = retry_info

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": [
            {
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "compression_ratio": segment.compression_ratio,
            }
            for segment in segments
        ],
        "asr_quality": "good",
        "asr_quality_reasons": [],
    }


def get_device() -> str:
    """Return the device used by the ASR model."""
    return DEVICE
