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


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
) -> Dict:
    """
    Transcribe an audio file using faster-whisper large-v3-turbo.

    Returns a dict compatible with the existing pipeline:
        {
            "text":                 str,
            "language":             str,
            "segments":             list[dict],
            "language_probability": float,
            "duration":             float,
            "avg_logprob":          float | None,
            "no_speech_prob":       float | None,
            "compression_ratio":    float | None,
            "asr_quality":          "good" | "low" | "failed",
            "asr_quality_reasons":  list[str],
        }
    """

    model = load_whisper_model()

    # Build transcription options
    options = {
        "beam_size": 5,
        "vad_filter": True,          # skip silent / non-speech regions
    }
    if language:
        options["language"] = language

    segments_gen, info = model.transcribe(
        str(audio_path),
        **options,
    )

    # Consume the generator and build serialisable segment dicts
    segment_list = []
    for seg in segments_gen:
        segment_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "compression_ratio": seg.compression_ratio,
        })

    # Full transcription text
    full_text = " ".join(
        s["text"] for s in segment_list if s["text"]
    )

    # Aggregate quality metrics across segments
    if segment_list:
        avg_logprob = (
            sum(s["avg_logprob"] for s in segment_list)
            / len(segment_list)
        )
        no_speech_prob = (
            sum(s["no_speech_prob"] for s in segment_list)
            / len(segment_list)
        )
        compression_ratio = (
            sum(s["compression_ratio"] for s in segment_list)
            / len(segment_list)
        )
    else:
        avg_logprob = None
        no_speech_prob = None
        compression_ratio = None

    result = {
        "text": full_text,
        "language": info.language,
        "segments": segment_list,
        # Quality / diagnostic fields
        "language_probability": info.language_probability,
        "duration": info.duration,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
    }

    # ASR quality validation
    quality = validate_asr_quality(result)
    result["asr_quality"] = quality["quality"]
    result["asr_quality_reasons"] = quality["reasons"]

    return result


def get_device() -> str:
    """Return the device used by the ASR model."""
    return DEVICE