from typing import Dict, List

from src.config import SUPPORTED_LANGUAGES
from src.language.segmenter import get_language_segments
from src.normalization.corrector import normalize_text
from src.semantic.interpreter import interpret_code_mix
from src.response.generator import generate_response
from src.response.voice_response import generate_voice_response
from src.voice.whisper_engine import transcribe_audio


# ============================================================
# PRIMARY LANGUAGE DERIVATION
# ============================================================

_SUPPORTED_NON_ENGLISH = [
    lang for lang in SUPPORTED_LANGUAGES
    if lang != "English"
]


def derive_primary_language(
    segments: List[Dict],
) -> str:
    """
    Determine the primary language from token-level segments.

    Rules:
    1. Ignore English when non-English supported languages
       are present.
    2. Weight each token by its confidence score.
    3. The supported non-English language with the highest
       weighted presence wins.
    4. If no non-English language is detected, fall back
       to English.

    This replaces the unreliable Whisper acoustic language
    guess for user-facing display.
    """

    weighted_counts: Dict[str, float] = {}

    for segment in segments:

        language = segment.get("language", "Unknown")
        confidence = segment.get("confidence", 0.0)

        if language == "Unknown":
            continue

        if language not in SUPPORTED_LANGUAGES:
            continue

        weighted_counts[language] = (
            weighted_counts.get(language, 0.0) + confidence
        )

    # Check for any non-English supported language.
    non_english = {
        lang: score
        for lang, score in weighted_counts.items()
        if lang in _SUPPORTED_NON_ENGLISH
    }

    if non_english:
        return max(non_english, key=non_english.get)

    # Fall back to English if present.
    if "English" in weighted_counts:
        return "English"

    return "Unknown"


def process_text(
    text: str,
    target_language: str = "",
) -> Dict:

    text = (text or "").strip()

    if not text:
        raise ValueError("Please enter some text.")

    segments = get_language_segments(text)

    normalized = normalize_text(
        text=text,
        segments=segments,
    )

    semantic = interpret_code_mix(
        text=text,
        segments=segments,
    )

    semantic_text = semantic.get(
        "semantic_text",
        text,
    )

    response = generate_response(
        text=text,
        segments=segments,
        semantic_input=semantic_text,
        max_new_tokens=100,
    )

    voice = generate_voice_response(
        text=text,
        segments=segments,
        semantic_input=semantic_text,
        response=response,
        target_language=target_language,
    )

    return {
        "input": text,
        "transcription": text,
        "segments": segments,
        "normalized": normalized,
        "semantic": semantic,
        "response": response,
        "response_language": voice["response_language"],
        "response_tts_text": voice["response_tts_text"],
        "response_audio_path": voice["audio_path"],
        "audio_error": voice["audio_error"],
        "status": "success",
    }


def process_voice(
    audio_path: str,
    target_language: str = "",
) -> Dict:

    if not audio_path:
        raise ValueError("Please provide an audio file.")

    transcription = transcribe_audio(
        audio_path,
    )

    text = transcription.get(
        "text",
        "",
    ).strip()

    if not text:
        raise ValueError("No speech detected.")

    result = process_text(
        text=text,
        target_language=target_language,
    )

    result["audio_input"] = audio_path
    result["whisper_language"] = transcription.get(
        "language",
        "unknown",
    )
    result["language_probability"] = transcription.get(
        "language_probability",
        0.0,
    )
    result["asr_quality"] = transcription.get(
        "asr_quality",
        "unknown",
    )
    result["asr_quality_reasons"] = transcription.get(
        "asr_quality_reasons",
        [],
    )

    # Derive the primary language from segmentation,
    # not from Whisper's acoustic guess.
    result["primary_language"] = derive_primary_language(
        result.get("segments", []),
    )

    return result

