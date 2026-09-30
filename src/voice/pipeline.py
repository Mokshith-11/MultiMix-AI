from typing import Dict

from src.language.segmenter import get_language_segments
from src.normalization.corrector import normalize_text
from src.semantic.interpreter import interpret_code_mix
from src.response.generator import generate_response
from src.voice.whisper_engine import transcribe_audio


# User-facing messages
_ASR_FAILED_MESSAGE = (
    "Sorry, I couldn't reliably understand the audio. "
    "Please try recording again with clearer speech."
)


def _empty_result(
    audio_path: str,
    transcription: str = "",
    whisper_language: str = "unknown",
    asr_quality: str = "failed",
    asr_quality_reasons: list = None,
    status: str = "",
) -> Dict:
    """Build a pipeline result when processing cannot continue."""
    return {
        "audio_path": audio_path,
        "transcription": transcription,
        "whisper_language": whisper_language,
        "asr_quality": asr_quality,
        "asr_quality_reasons": asr_quality_reasons or [],
        "segments": [],
        "normalized": {},
        "semantic": {},
        "response": "",
        "response_error": None,
        "status": status,
    }


def process_voice(audio_path: str) -> Dict:
    """
    Complete MultiMix AI voice pipeline.

    Audio
        ↓
    ASR (faster-whisper)
        ↓
    ASR quality validation       ← NEW (Task 3)
        ↓
    Language segmentation
        ↓
    Context resolution
        ↓
    Normalization
        ↓
    Semantic interpretation
        ↓
    AI response generation
    """

    # 1. ASR transcription
    transcription = transcribe_audio(
        audio_path=audio_path
    )

    text = transcription.get("text", "").strip()
    whisper_language = transcription.get(
        "language",
        "unknown"
    )
    asr_quality = transcription.get("asr_quality", "good")
    asr_quality_reasons = transcription.get(
        "asr_quality_reasons", []
    )

    # 2. No speech detected
    if not text:
        return _empty_result(
            audio_path=audio_path,
            whisper_language=whisper_language,
            asr_quality="failed",
            asr_quality_reasons=["No speech detected"],
            status="No speech detected",
        )

    # 3. ASR quality gate — block clearly unusable transcriptions
    if asr_quality == "failed":
        return _empty_result(
            audio_path=audio_path,
            transcription=text,
            whisper_language=whisper_language,
            asr_quality="failed",
            asr_quality_reasons=asr_quality_reasons,
            status=_ASR_FAILED_MESSAGE,
        )

    # 4. Language segmentation
    segments = get_language_segments(text)

    # 5. Text normalization
    normalized = normalize_text(
        text=text,
        segments=segments
    )

    # 6. Semantic interpretation
    semantic = interpret_code_mix(
        text=text,
        segments=segments
    )

    # 7. AI response generation
    response = ""
    response_error = None

    try:
        response = generate_response(
            text=text,
            segments=segments,
            semantic_input=semantic.get(
                "semantic_text",
                text
            ),
            max_new_tokens=100
        )

    except Exception as error:
        response_error = str(error)

    # 8. Determine final status
    if asr_quality == "low":
        status = "success_low_confidence"
    else:
        status = "success"

    # 9. Final result
    return {
        "audio_path": audio_path,
        "transcription": text,
        "whisper_language": whisper_language,
        "asr_quality": asr_quality,
        "asr_quality_reasons": asr_quality_reasons,
        "segments": segments,
        "normalized": normalized,
        "semantic": semantic,
        "response": response,
        "response_error": response_error,
        "status": status,
    }