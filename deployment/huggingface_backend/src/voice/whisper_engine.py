from typing import Dict, Optional

from faster_whisper import WhisperModel

from src.config import WHISPER_MODEL_ID

_model: Optional[WhisperModel] = None


def load_whisper_model():
    global _model

    if _model is not None:
        return _model

    print("Loading faster-whisper large-v3-turbo...")

    _model = WhisperModel(
        WHISPER_MODEL_ID,
        device="cuda",
        compute_type="float16",
    )

    print("Whisper loaded.")
    return _model


def get_device() -> str:
    return "cuda"


def validate_asr_quality(transcription: Dict) -> Dict:
    text = transcription.get("text", "").strip()
    avg_logprob = transcription.get("avg_logprob")
    no_speech_prob = transcription.get("no_speech_prob")
    compression_ratio = transcription.get("compression_ratio")

    if not text:
        return {
            "quality": "failed",
            "reasons": ["Empty transcription"],
        }

    reasons = []

    if avg_logprob is not None and avg_logprob < -1.5:
        return {
            "quality": "failed",
            "reasons": [f"Very low avg_logprob: {avg_logprob:.3f}"],
        }

    if no_speech_prob is not None and no_speech_prob > 0.8:
        return {
            "quality": "failed",
            "reasons": [f"High no_speech_prob: {no_speech_prob:.3f}"],
        }

    if avg_logprob is not None and avg_logprob < -0.7:
        reasons.append("Lower transcription confidence")

    if no_speech_prob is not None and no_speech_prob > 0.3:
        reasons.append("Elevated no-speech probability")

    if compression_ratio is not None and compression_ratio > 2.4:
        reasons.append("High compression ratio")

    return {
        "quality": "good" if not reasons else "low",
        "reasons": reasons,
    }


def transcribe_audio(audio_path: str) -> Dict:
    model = load_whisper_model()

    romanization_prompt = (
        "Romanized transcription only. "
        "Use Latin alphabet. "
        "Do not translate. "
        "Do not use Devanagari, Telugu, Tamil, Bengali or other Indic scripts. "
        "Preserve Telugu, Tamil, Hindi, Bengali and English words as spoken. "
        "Example: Nenu today college ki vellanu but my friend "
        "Tamil-la pesitu irindhan."
    )

    segments, info = model.transcribe(
        audio_path,
        task="transcribe",
        beam_size=5,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=romanization_prompt,
    )

    segments = list(segments)

    text = " ".join(
        s.text.strip()
        for s in segments
    ).strip()

    has_indic = any(
        "\u0900" <= ch <= "\u097F" or
        "\u0C00" <= ch <= "\u0C7F" or
        "\u0B80" <= ch <= "\u0BFF" or
        "\u0980" <= ch <= "\u09FF"
        for ch in text
    )

    # One retry if Whisper emits Indic script.
    if has_indic:
        retry_prompt = (
            "WRITE ONLY IN LATIN LETTERS. "
            "ROMANIZED TRANSCRIPTION ONLY. "
            "DO NOT TRANSLATE. "
            "Example: Nenu today college ki vellanu but my friend "
            "Tamil-la pesitu irindhan."
        )

        retry_segments, retry_info = model.transcribe(
            audio_path,
            task="transcribe",
            beam_size=5,
            temperature=0,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=retry_prompt,
        )

        retry_segments = list(retry_segments)

        retry_text = " ".join(
            s.text.strip()
            for s in retry_segments
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
            segments = retry_segments

    if segments:
        first = segments[0]
        diagnostics = {
            "avg_logprob": first.avg_logprob,
            "no_speech_prob": first.no_speech_prob,
            "compression_ratio": first.compression_ratio,
        }
    else:
        diagnostics = {}

    quality = validate_asr_quality(
        {
            "text": text,
            **diagnostics,
        }
    )

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": [
            {
                "text": s.text.strip(),
                "start": s.start,
                "end": s.end,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "compression_ratio": s.compression_ratio,
            }
            for s in segments
        ],
        "asr_quality": quality["quality"],
        "asr_quality_reasons": quality["reasons"],
    }
