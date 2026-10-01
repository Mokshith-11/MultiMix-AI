from typing import Dict

from src.language.segmenter import get_language_segments
from src.normalization.corrector import normalize_text
from src.semantic.interpreter import interpret_code_mix
from src.response.generator import generate_response
from src.response.voice_response import generate_voice_response
from src.voice.whisper_engine import transcribe_audio


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

    return result
