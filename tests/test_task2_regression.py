"""
Task 2 regression test — faster-whisper integration.

Tests:
  1. whisper_engine.transcribe_audio() directly on both audio files
  2. Full voice pipeline (ASR → segmenter → normalizer → semantic → Qwen)
  3. Text pipeline only (segmenter → normalizer → semantic → Qwen)
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

AUDIO_FILES = [
    PROJECT_ROOT / "assets" / "audio" / "WhatsApp Audio 2026-09-29 at 12.29.00 PM.mp4",
    PROJECT_ROOT / "assets" / "audio" / "WhatsApp Audio 2026-09-29 at 12.32.26 PM.mp4",
]

TEXT_TESTS = [
    "Nenu today college ki vellanu, but my friend Tamil-la pesitu irundhan",
    "What is 2 plus 2?",
]


def divider(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ------------------------------------------------------------------
# Test 1: whisper_engine standalone
# ------------------------------------------------------------------
def test_whisper_engine():
    divider("TEST 1: whisper_engine.transcribe_audio()")

    from multimix_src.voice.whisper_engine import transcribe_audio, load_whisper_model, get_device

    print(f"Device: {get_device()}")

    # Ensure model loads without error
    model = load_whisper_model()
    print(f"Model loaded: {type(model).__name__}")

    for audio_path in AUDIO_FILES:
        print(f"\n--- {audio_path.name} ---")
        t0 = time.time()
        result = transcribe_audio(audio_path=str(audio_path))
        elapsed = time.time() - t0

        print(f"  Language       : {result['language']}")
        print(f"  Language prob  : {result.get('language_probability', 'N/A')}")
        print(f"  Text           : {result['text']}")
        print(f"  Segments       : {len(result['segments'])}")
        print(f"  avg_logprob    : {result.get('avg_logprob')}")
        print(f"  no_speech_prob : {result.get('no_speech_prob')}")
        print(f"  Duration (s)   : {result.get('duration')}")
        print(f"  Processing (s) : {elapsed:.2f}")

        # Basic sanity checks
        assert result["text"], "Transcription should not be empty"
        assert result["language"], "Language should be detected"
        assert len(result["segments"]) > 0, "Should have at least one segment"
        print("  STATUS: PASS")

    print("\nTEST 1: ALL PASS")


# ------------------------------------------------------------------
# Test 2: Text pipeline (no audio)
# ------------------------------------------------------------------
def test_text_pipeline():
    divider("TEST 2: Text pipeline (segmenter + normalizer + semantic + Qwen)")

    from multimix_src.language.segmenter import get_language_segments
    from multimix_src.normalization.corrector import normalize_text
    from multimix_src.semantic.interpreter import interpret_code_mix
    from multimix_src.response.generator import generate_response

    for text in TEXT_TESTS:
        print(f"\n--- Input: {text} ---")

        segments = get_language_segments(text)
        langs = list({s.get("language", "unknown") for s in segments})
        print(f"  Languages : {langs}")

        normalized = normalize_text(text=text, segments=segments)
        print(f"  Normalized: {normalized}")

        semantic = interpret_code_mix(text=text, segments=segments)
        semantic_text = semantic.get("semantic_text", text)
        print(f"  Semantic  : {semantic_text}")

        response = generate_response(
            text=text,
            segments=segments,
            semantic_input=semantic_text,
            max_new_tokens=100,
        )
        print(f"  Qwen resp : {response[:120]}...")

        assert response, "Qwen should generate a response"
        print("  STATUS: PASS")

    print("\nTEST 2: ALL PASS")


# ------------------------------------------------------------------
# Test 3: Full voice pipeline
# ------------------------------------------------------------------
def test_voice_pipeline():
    divider("TEST 3: Full voice pipeline (process_voice)")

    from multimix_src.voice.pipeline import process_voice

    for audio_path in AUDIO_FILES:
        print(f"\n--- {audio_path.name} ---")
        t0 = time.time()
        result = process_voice(audio_path=str(audio_path))
        elapsed = time.time() - t0

        print(f"  Status        : {result.get('status')}")
        print(f"  Transcription : {result.get('transcription')}")
        print(f"  Language      : {result.get('whisper_language')}")
        print(f"  Segments      : {len(result.get('segments', []))}")
        print(f"  Semantic      : {result.get('semantic', {}).get('semantic_text', 'N/A')}")
        print(f"  Response      : {str(result.get('response', ''))[:120]}...")
        print(f"  Resp error    : {result.get('response_error')}")
        print(f"  Total time    : {elapsed:.2f}s")

        assert result.get("status") == "success", f"Pipeline status should be success, got: {result.get('status')}"
        assert result.get("transcription"), "Transcription should not be empty"
        assert result.get("response") or result.get("response_error") is None, "Should have response or no error"
        print("  STATUS: PASS")

    print("\nTEST 3: ALL PASS")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("MultiMix AI — Task 2 Regression Test")
    print(f"Project root: {PROJECT_ROOT}")

    test_whisper_engine()
    test_text_pipeline()
    test_voice_pipeline()

    divider("ALL TESTS COMPLETE")
    print("Task 2 integration verified successfully.")
