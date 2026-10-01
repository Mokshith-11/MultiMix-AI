import uuid
from typing import Dict

from src.config import GENERATED_AUDIO_DIR
from src.response.generator import generate_response
from src.voice.tts_engine import synthesize_speech
from src.response.translation import translate_to_indic


def generate_voice_response(
    text: str,
    segments,
    semantic_input: str,
    response: str = "",
    target_language: str = "",
) -> Dict:
    """
    Generate a grounded AI response and convert it into
    an Indian-language voice response using IndicF5.

    Parameters
    ----------
    text:
        Original user input.

    segments:
        Detected language segments.

    semantic_input:
        Semantic interpretation of the user input.

    response:
        Optional already-generated Qwen response.
        Providing this prevents duplicate Qwen inference.

    target_language:
        Optional output language. When empty, the dominant
        supported Indian language in the input is selected.
    """

    # ---------------------------------------------------------
    # Step 1: Generate AI response only when not already given
    # ---------------------------------------------------------

    if not response.strip():
        response = generate_response(
            text=text,
            segments=segments,
            semantic_input=semantic_input,
            max_new_tokens=80,
        )

    if not response:
        raise RuntimeError(
            "Qwen returned an empty response."
        )

    # ---------------------------------------------------------
    # Step 2: Determine output language
    # ---------------------------------------------------------

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

    if target_language.strip():
        response_language = target_language.strip()
    elif language_counts:
        response_language = max(
            language_counts,
            key=language_counts.get,
        )
    else:
        response_language = "English"

    # ---------------------------------------------------------
    # Step 3: Translate Qwen response for IndicF5
    # ---------------------------------------------------------

    response_for_tts = response
    audio_path = None
    audio_error = None

    if response_language.lower() not in {
        "english",
        "en",
    }:
        try:
            response_for_tts = translate_to_indic(
                response,
                response_language,
            )

        except Exception as error:
            audio_error = (
                f"Response translation failed: {error}"
            )

    else:
        # IndicF5 is intended for Indian-language synthesis.
        audio_error = (
            "IndicF5 voice output is not used for English-only "
            "responses. Select Telugu, Tamil, Hindi or Bengali "
            "for Indian-language voice output."
        )

    # ---------------------------------------------------------
    # Step 4: Generate IndicF5 speech
    # ---------------------------------------------------------

    if not audio_error:
        output_dir = GENERATED_AUDIO_DIR

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        generated_path = (
            output_dir
            / f"response_{uuid.uuid4().hex}.wav"
        )

        try:
            audio_path = synthesize_speech(
                text=response_for_tts,
                output_path=str(generated_path),
                speed=1.0,
            )

        except Exception as error:
            audio_error = (
                f"IndicF5 synthesis failed: {error}"
            )

    # ---------------------------------------------------------
    # Step 5: Return complete result
    # ---------------------------------------------------------

    return {
        "input": text,
        "semantic": semantic_input,
        "response": response,
        "response_language": response_language,
        "response_tts_text": response_for_tts,
        "audio_path": audio_path,
        "audio_error": audio_error,
    }
