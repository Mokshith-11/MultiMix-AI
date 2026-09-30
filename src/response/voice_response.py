import uuid
from typing import Dict

from src.config import GENERATED_AUDIO_DIR
from src.response.generator import generate_response
from src.voice.tts_engine import synthesize_speech


def generate_voice_response(
    text: str,
    segments,
    semantic_input: str,
) -> Dict:
    """
    Complete MultiMix response pipeline.
    """

    # ---------------------------------------------------------
    # Step 1: Generate AI response
    # ---------------------------------------------------------

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
    # Step 2: Generate speech from AI response
    # ---------------------------------------------------------

    output_dir = GENERATED_AUDIO_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_path = (
        output_dir
        / f"response_{uuid.uuid4().hex}.wav"
    )

    generated_audio = synthesize_speech(
        text=response,
        output_path=str(audio_path),
        speed=1.0,
    )

    # ---------------------------------------------------------
    # Step 3: Return complete result
    # ---------------------------------------------------------

    return {
        "input": text,
        "semantic": semantic_input,
        "response": response,
        "audio_path": generated_audio,
    }