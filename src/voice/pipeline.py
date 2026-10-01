from typing import Dict

from src.language.segmenter import get_language_segments
from src.normalization.corrector import normalize_text
from src.semantic.interpreter import interpret_code_mix
from src.response.generator import generate_response
from src.response.voice_response import generate_voice_response
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
    ASR quality validation
        ↓
    Language segmentation
        ↓
    Normalization
        ↓
    Semantic interpretation
        ↓
    Qwen response
        ↓
    Target-language translation
        ↓
    IndicF5 voice response
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
    asr_quality = transcription.get(
        "asr_quality",
        "good"
    )
    asr_quality_reasons = transcription.get(
        "asr_quality_reasons",
        []
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

    # 3. ASR quality gate
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

    semantic_text = semantic.get(
        "semantic_text",
        text
    )

    # 7. AI response generation
    response = ""
    response_error = None

    try:
        response = generate_response(
            text=text,
            segments=segments,
            semantic_input=semantic_text,
            max_new_tokens=100
        )

    except Exception as error:
        response_error = str(error)

    # 8. Generate translated Indian-language voice
    audio_path_result = None
    response_language = "English"
    response_tts_text = ""
    audio_error = None

    if response:
        try:
            # Determine the dominant supported Indian language
            # in the user's code-mixed input.
            language_counts = {}

            for segment in segments:
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

            if language_counts:
                response_language = max(
                    language_counts,
                    key=language_counts.get,
                )

            voice_result = generate_voice_response(
                text=text,
                segments=segments,
                semantic_input=semantic_text,
                response=response,
                target_language=response_language,
            )

            audio_path_result = voice_result.get(
                "audio_path"
            )
            response_language = voice_result.get(
                "response_language",
                response_language,
            )
            response_tts_text = voice_result.get(
                "response_tts_text",
                "",
            )
            audio_error = voice_result.get(
                "audio_error"
            )

        except Exception as error:
            audio_error = str(error)

    # 9. Final status
    if asr_quality == "low":
        status = "success_low_confidence"
    else:
        status = "success"

    # 10. Final result
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

        # Final AI voice response
        "response_language": response_language,
        "response_tts_text": response_tts_text,
        "response_audio_path": audio_path_result,
        "audio_error": audio_error,

        "status": status,
    }
