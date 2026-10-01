import uuid
from typing import Dict

from src.config import GENERATED_AUDIO_DIR
from src.response.generator import generate_response
from src.response.translation import translate_to_indic
from src.voice.tts_engine import synthesize_speech


def generate_voice_response(
    text: str,
    segments,
    semantic_input: str,
    response: str = "",
    target_language: str = "",
) -> Dict:

    if not response.strip():
        response = generate_response(
            text=text,
            segments=segments,
            semantic_input=semantic_input,
            max_new_tokens=80,
        )

    if not response:
        raise RuntimeError("Empty Qwen response.")

    language_counts = {}

    for segment in segments or []:
        language = segment.get("language", "")

        if language in {
            "Telugu",
            "Tamil",
            "Hindi",
            "Bengali",
        }:
            language_counts[language] = (
                language_counts.get(language, 0) + 1
            )

    if target_language and target_language.strip():
        response_language = target_language.strip()
    elif language_counts:
        response_language = max(
            language_counts,
            key=language_counts.get,
        )
    else:
        response_language = "English"

    response_for_tts = response
    audio_path = None
    audio_error = None

    if response_language.lower() not in {"english", "en"}:
        try:
            response_for_tts = translate_to_indic(
                response,
                response_language,
            )

            generated_path = (
                GENERATED_AUDIO_DIR
                / f"response_{uuid.uuid4().hex}.wav"
            )

            audio_path = synthesize_speech(
                text=response_for_tts,
                output_path=str(generated_path),
            )

        except Exception as error:
            audio_error = str(error)

    else:
        audio_error = (
            "IndicF5 voice output is intended for Indian-language responses."
        )

    return {
        "input": text,
        "semantic": semantic_input,
        "response": response,
        "response_language": response_language,
        "response_tts_text": response_for_tts,
        "audio_path": audio_path,
        "audio_error": audio_error,
    }
