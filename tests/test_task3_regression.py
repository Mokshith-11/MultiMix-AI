"""
Task 3 regression test — ASR Quality Validation.

Tests:
  1. whisper_engine quality fields on both audio files
  2. validate_asr_quality() with synthetic good/low/failed inputs
  3. Full voice pipeline with quality gate
  4. Text pipeline still works (Qwen not broken)
  5. Simulated failed ASR produces user-friendly response
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

passed = 0
failed = 0


def report(test_name: str, ok: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))


def divider(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ------------------------------------------------------------------
# Test 1: whisper_engine returns quality fields
# ------------------------------------------------------------------
def test_whisper_engine_quality():
    divider("TEST 1: whisper_engine quality fields")

    from src.voice.whisper_engine import transcribe_audio

    for audio_path in AUDIO_FILES:
        print(f"\n  File: {audio_path.name}")
        result = transcribe_audio(audio_path=str(audio_path))

        # Check quality fields exist
        report(
            f"{audio_path.name} has asr_quality",
            "asr_quality" in result,
            f"asr_quality={result.get('asr_quality')}"
        )
        report(
            f"{audio_path.name} has asr_quality_reasons",
            "asr_quality_reasons" in result,
            f"reasons={result.get('asr_quality_reasons')}"
        )
        report(
            f"{audio_path.name} quality is good",
            result.get("asr_quality") == "good",
            f"These real audio files should be 'good'"
        )

        # Print diagnostics
        print(f"    text         : {result['text'][:80]}...")
        print(f"    avg_logprob  : {result.get('avg_logprob')}")
        print(f"    no_speech    : {result.get('no_speech_prob')}")
        print(f"    compression  : {result.get('compression_ratio')}")


# ------------------------------------------------------------------
# Test 2: validate_asr_quality() unit tests
# ------------------------------------------------------------------
def test_validate_asr_quality_unit():
    divider("TEST 2: validate_asr_quality() unit tests")

    from src.voice.whisper_engine import validate_asr_quality

    # Good transcription
    good = validate_asr_quality({
        "text": "Hello world this is a test",
        "avg_logprob": -0.15,
        "no_speech_prob": 0.001,
        "compression_ratio": 1.2,
    })
    report("Good input → good", good["quality"] == "good", f"got: {good}")

    # Low quality — marginal avg_logprob
    low = validate_asr_quality({
        "text": "Some uncertain text",
        "avg_logprob": -0.9,
        "no_speech_prob": 0.05,
        "compression_ratio": 1.5,
    })
    report("Marginal logprob → low", low["quality"] == "low", f"got: {low}")

    # Low quality — elevated no_speech_prob
    low2 = validate_asr_quality({
        "text": "Maybe speech maybe not",
        "avg_logprob": -0.3,
        "no_speech_prob": 0.5,
        "compression_ratio": 1.3,
    })
    report("Elevated no_speech → low", low2["quality"] == "low", f"got: {low2}")

    # Low quality — high compression ratio (repetitive)
    low3 = validate_asr_quality({
        "text": "test test test test test",
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
        "compression_ratio": 3.0,
    })
    report("High compression → low", low3["quality"] == "low", f"got: {low3}")

    # Failed — empty text
    fail_empty = validate_asr_quality({
        "text": "",
        "avg_logprob": -0.1,
        "no_speech_prob": 0.0,
        "compression_ratio": 1.0,
    })
    report("Empty text → failed", fail_empty["quality"] == "failed", f"got: {fail_empty}")

    # Failed — terrible avg_logprob (like old Whisper Small garbage)
    fail_logprob = validate_asr_quality({
        "text": "garbled garbage text",
        "avg_logprob": -1.88,
        "no_speech_prob": 0.1,
        "compression_ratio": 1.5,
    })
    report("Terrible logprob → failed", fail_logprob["quality"] == "failed", f"got: {fail_logprob}")

    # Failed — very high no_speech_prob
    fail_nospeech = validate_asr_quality({
        "text": "phantom text",
        "avg_logprob": -0.3,
        "no_speech_prob": 0.95,
        "compression_ratio": 1.2,
    })
    report("Very high no_speech → failed", fail_nospeech["quality"] == "failed", f"got: {fail_nospeech}")

    # Good — no metrics available (None values should not crash)
    none_metrics = validate_asr_quality({
        "text": "Some text with no metrics",
        "avg_logprob": None,
        "no_speech_prob": None,
        "compression_ratio": None,
    })
    report("None metrics → good (no signal to reject)", none_metrics["quality"] == "good", f"got: {none_metrics}")


# ------------------------------------------------------------------
# Test 3: Pipeline returns asr_quality for real audio
# ------------------------------------------------------------------
def test_pipeline_with_quality():
    divider("TEST 3: Full voice pipeline with ASR quality")

    from src.voice.pipeline import process_voice

    for audio_path in AUDIO_FILES:
        print(f"\n  File: {audio_path.name}")
        result = process_voice(audio_path=str(audio_path))

        report(
            f"{audio_path.name} has asr_quality in result",
            "asr_quality" in result,
            f"asr_quality={result.get('asr_quality')}"
        )
        report(
            f"{audio_path.name} status is success*",
            result.get("status", "").startswith("success"),
            f"status={result.get('status')}"
        )
        report(
            f"{audio_path.name} has response",
            bool(result.get("response")),
            f"response={str(result.get('response', ''))[:80]}..."
        )

        print(f"    transcription: {result.get('transcription', '')[:80]}...")
        print(f"    asr_quality  : {result.get('asr_quality')}")
        print(f"    asr_reasons  : {result.get('asr_quality_reasons')}")


# ------------------------------------------------------------------
# Test 4: Text pipeline (Qwen) still works
# ------------------------------------------------------------------
def test_text_pipeline():
    divider("TEST 4: Text pipeline (Qwen) still works")

    from src.language.segmenter import get_language_segments
    from src.normalization.corrector import normalize_text
    from src.semantic.interpreter import interpret_code_mix
    from src.response.generator import generate_response

    text = "Nenu today college ki vellanu, but my friend Tamil-la pesitu irundhan"
    segments = get_language_segments(text)
    langs = list({s.get("language", "unknown") for s in segments})
    report("Segmenter works", len(segments) > 0, f"langs={langs}")

    normalized = normalize_text(text=text, segments=segments)
    report("Normalizer works", bool(normalized), "")

    semantic = interpret_code_mix(text=text, segments=segments)
    semantic_text = semantic.get("semantic_text", text)
    report("Semantic works", bool(semantic_text), f"semantic={semantic_text[:60]}")

    response = generate_response(
        text=text,
        segments=segments,
        semantic_input=semantic_text,
        max_new_tokens=100,
    )
    report("Qwen responds", bool(response), f"response={response[:80]}...")


# ------------------------------------------------------------------
# Test 5: Simulated failed ASR is handled gracefully
# ------------------------------------------------------------------
def test_failed_asr_handling():
    divider("TEST 5: Failed ASR handling in pipeline")

    from src.voice.pipeline import _empty_result, _ASR_FAILED_MESSAGE

    # Simulate what happens when ASR quality is "failed"
    result = _empty_result(
        audio_path="fake_audio.mp4",
        transcription="garbled nonsense",
        whisper_language="unknown",
        asr_quality="failed",
        asr_quality_reasons=["Very low avg_logprob (-1.880 < -1.5)"],
        status=_ASR_FAILED_MESSAGE,
    )

    report("Failed result has no response sent to Qwen", result["response"] == "", "")
    report("Failed result has friendly status", "couldn't reliably" in result["status"], f"status={result['status']}")
    report("Failed result preserves transcription for debugging", result["transcription"] == "garbled nonsense", "")
    report("Failed result has asr_quality=failed", result["asr_quality"] == "failed", "")
    report("Failed result has empty segments", result["segments"] == [], "")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("MultiMix AI — Task 3 Regression Test")
    print(f"Project root: {PROJECT_ROOT}")

    test_validate_asr_quality_unit()   # Fast, no model needed
    test_failed_asr_handling()         # Fast, no model needed
    test_whisper_engine_quality()      # Needs faster-whisper model
    test_text_pipeline()               # Needs Qwen
    test_pipeline_with_quality()       # Needs both

    divider("SUMMARY")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    if failed == 0:
        print("\n  ALL TESTS PASSED ✓")
    else:
        print(f"\n  {failed} TEST(S) FAILED ✗")
        sys.exit(1)
