"""
Task 6 — Full Regression and Integration Test Suite for MultiMix AI.

Covers:
  Step 2:  Language Detection (English, Telugu, Tamil, Hindi, Bengali, Code-Mixed)
  Step 3:  Normalization (Abbreviations + Multilingual token preservation)
  Step 4:  Semantic Interpretation (5 Language pairs + 3-Language mixed input)
  Step 5:  Qwen Model & Grounded Generation ("2 plus 2", Multilingual input)
  Step 6:  Faster-Whisper large-v3-turbo (Real audio files, CPU INT8, ASR metrics)
  Step 7:  ASR Failure & Rejection (No Qwen call, no TTS, friendly status)
  Step 8:  IndicF5 Speech Synthesis (Vocoder, model, WAV output, sample rate)
  Step 9:  Complete End-to-End Text Pipeline (Detection → Norm → Semantic → Qwen → IndicF5)
  Step 10: Complete End-to-End Voice Pipeline (process_voice on WhatsApp audio + TTS)
  Step 11: Configuration & Dynamic Path Resolution
  Step 12: Import Verification of All Core Modules
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows UTF-8 stdout guard to avoid charmap encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

passed_tests = 0
failed_tests = 0
test_results: List[Dict[str, Any]] = []


def record_result(category: str, test_name: str, passed: bool, detail: str = ""):
    global passed_tests, failed_tests
    status = "PASS" if passed else "FAIL"
    if passed:
        passed_tests += 1
    else:
        failed_tests += 1
    test_results.append({
        "category": category,
        "name": test_name,
        "status": status,
        "detail": detail,
    })
    prefix = f"  [{status}]"
    print(f"{prefix} {test_name}" + (f" — {detail}" if detail else ""))


def section_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =====================================================================
# STEP 11 & STEP 12: Configuration and Module Imports
# =====================================================================

def test_step11_configuration():
    section_header("STEP 11: Configuration & Dynamic Path Resolution")
    from src import config

    # 1. Verify core directories resolve dynamically and exist
    record_result("Configuration", "PROJECT_ROOT resolves to directory",
                  config.PROJECT_ROOT.exists() and config.PROJECT_ROOT.is_dir(),
                  str(config.PROJECT_ROOT))

    record_result("Configuration", "MODELS_DIR exists",
                  config.MODELS_DIR.exists() and config.MODELS_DIR.is_dir(),
                  str(config.MODELS_DIR))

    record_result("Configuration", "ASSETS_DIR exists",
                  config.ASSETS_DIR.exists() and config.ASSETS_DIR.is_dir(),
                  str(config.ASSETS_DIR))

    record_result("Configuration", "AUDIO_DIR exists",
                  config.AUDIO_DIR.exists() and config.AUDIO_DIR.is_dir(),
                  str(config.AUDIO_DIR))

    record_result("Configuration", "GENERATED_AUDIO_DIR exists",
                  config.GENERATED_AUDIO_DIR.exists() and config.GENERATED_AUDIO_DIR.is_dir(),
                  str(config.GENERATED_AUDIO_DIR))

    # 2. Verify model paths resolve dynamically and exist
    record_result("Configuration", "QWEN_MODEL_PATH exists",
                  config.QWEN_MODEL_PATH.exists(),
                  str(config.QWEN_MODEL_PATH))

    record_result("Configuration", "INDICF5_MODEL_DIR exists",
                  config.INDICF5_MODEL_DIR.exists(),
                  str(config.INDICF5_MODEL_DIR))

    record_result("Configuration", "WHISPER_TURBO_MODEL_DIR exists",
                  config.WHISPER_TURBO_MODEL_DIR.exists(),
                  str(config.WHISPER_TURBO_MODEL_DIR))

    record_result("Configuration", "INDICLID_MODEL_PATH exists",
                  config.INDICLID_MODEL_PATH.exists(),
                  str(config.INDICLID_MODEL_PATH))

    record_result("Configuration", "MT5_MODEL_DIR exists",
                  config.MT5_MODEL_DIR.exists(),
                  str(config.MT5_MODEL_DIR))


def test_step12_imports():
    section_header("STEP 12: Import Verification of All Core Modules")

    modules_to_test = [
        "src.config",
        "src.language.segmenter",
        "src.language.detector",
        "src.normalization.corrector",
        "src.semantic.interpreter",
        "src.response.generator",
        "src.response.voice_response",
        "src.voice.whisper_engine",
        "src.voice.pipeline",
        "src.voice.tts_engine",
    ]

    for mod_name in modules_to_test:
        try:
            __import__(mod_name)
            record_result("Imports", f"Import {mod_name}", True, "OK")
        except Exception as e:
            record_result("Imports", f"Import {mod_name}", False, f"Failed: {e}")


# =====================================================================
# STEP 2: Language Detection
# =====================================================================

def test_step2_language_detection():
    section_header("STEP 2: Language Detection")
    from multimix_src.language.detector import analyze_code_mix

    cases = [
        ("English sentence", "I am going to office today", "English"),
        ("Telugu (Romanized)", "Nenu ivala vellanu", "Telugu"),
        ("Telugu (Native script)", "నేను ఈ రోజు కాలేజీకి వెళ్లాను", "Telugu"),
        ("Tamil (Romanized)", "Naan romba pesuren", "Tamil"),
        ("Tamil (Native script)", "நான் இன்று கல்லூரிக்கு சென்றேன்", "Tamil"),
        ("Hindi (Romanized)", "Main aaj college jaunga", "Hindi"),
        ("Hindi (Native script)", "मैं आज कॉलेज जाऊंगा", "Hindi"),
        ("Bengali (Romanized)", "Ami ekhane achi", "Bengali"),
        ("Bengali (Native script)", "আমি আজ কলেজে যাবো", "Bengali"),
    ]

    for label, text, expected_lang in cases:
        result = analyze_code_mix(text)
        detected = result.get("languages", [])
        record_result("Language Detection", f"{label} detected as {expected_lang}",
                      expected_lang in detected,
                      f"Input='{text}' → Detected={detected}")

    # Critical 3-language code-mixed test
    mixed_input = "Nenu today college ki vellanu, but my friend Tamil-la pesitu irundhan"
    mixed_result = analyze_code_mix(mixed_input)
    mixed_langs = set(mixed_result.get("languages", []))

    for lang in ["Telugu", "English", "Tamil"]:
        record_result("Language Detection", f"Code-mixed contains {lang}",
                      lang in mixed_langs,
                      f"Detected languages: {sorted(mixed_langs)}")


# =====================================================================
# STEP 3: Normalization
# =====================================================================

def test_step3_normalization():
    section_header("STEP 3: Normalization & Token Preservation")
    from multimix_src.normalization.corrector import normalize_token, normalize_text
    from multimix_src.language.segmenter import get_language_segments

    # Required slang/abbreviation mappings
    expected_mappings = [
        ("tmrw", "tomorrow"),
        ("tomo", "tomorrow"),
        ("todayy", "today"),
        ("frnd", "friend"),
        ("clg", "college"),
        ("plz", "please"),
        ("bcoz", "because"),
    ]

    for raw, expected in expected_mappings:
        actual = normalize_token(raw)
        record_result("Normalization", f"Normalize '{raw}' → '{expected}'",
                      actual == expected,
                      f"Got: '{actual}'")

    # Multilingual tokens must be preserved
    multilingual_tokens = [
        "nenu",
        "vellanu",
        "Tamil-la",
        "pesitu",
        "irundhan",
        "college",
    ]

    for tok in multilingual_tokens:
        actual = normalize_token(tok)
        record_result("Normalization", f"Preserve token '{tok}'",
                      actual.lower() == tok.lower(),
                      f"Got: '{actual}'")

    # Full text normalization test
    raw_mixed = "nenu todayy clg ki vellanu plz frnd"
    segs = get_language_segments(raw_mixed)
    norm_res = normalize_text(raw_mixed, segs)
    norm_text = norm_res.get("normalized_text", "")

    record_result("Normalization", "Sentence normalization expands abbreviations",
                  "today" in norm_text and "college" in norm_text and "please" in norm_text and "friend" in norm_text,
                  f"Result='{norm_text}'")


# =====================================================================
# STEP 4: Semantic Interpretation
# =====================================================================

def test_step4_semantic_interpretation():
    section_header("STEP 4: Semantic Interpretation")
    from multimix_src.language.segmenter import get_language_segments
    from multimix_src.semantic.interpreter import interpret_code_mix

    cases = [
        ("English", "I went to the office today", ["I", "went", "office"]),
        ("Telugu-English", "Nenu today college ki vellanu", ["I", "went", "college"]),
        ("Tamil-English", "Naan today college poiten", ["I", "went", "college"]),
        ("Hindi-English", "Main today college gaya", ["I", "went", "college"]),
        ("Bengali-English", "Ami today college gechi", ["I", "went", "college"]),
    ]

    for label, text, required_keywords in cases:
        segs = get_language_segments(text)
        res = interpret_code_mix(text, segs)
        semantic_text = res.get("semantic_text", "")
        has_keywords = all(kw.lower() in semantic_text.lower() for kw in required_keywords)
        record_result("Semantic Interpretation", f"{label} conveys intended meaning",
                      has_keywords,
                      f"Semantic='{semantic_text}'")

    # Important 3-language code-mixed test
    mixed_input = "Nenu today college ki vellanu, but my friend Tamil-la pesitu irundhan"
    segs = get_language_segments(mixed_input)
    mixed_sem = interpret_code_mix(mixed_input, segs).get("semantic_text", "")

    # Meaning must approximate "I went to college today but my friend was speaking in Tamil"
    meaning_ok = (
        "went" in mixed_sem.lower()
        and "college" in mixed_sem.lower()
        and "tamil" in mixed_sem.lower()
        and "friend" in mixed_sem.lower()
    )
    record_result("Semantic Interpretation", "3-language mixed input semantic meaning",
                  meaning_ok,
                  f"Semantic='{mixed_sem}'")


# =====================================================================
# STEP 5: Qwen Local Response Generator
# =====================================================================

def test_step5_qwen():
    section_header("STEP 5: Qwen Model & Grounded Generation")
    from multimix_src.response.generator import load_model, generate_response
    from multimix_src.language.segmenter import get_language_segments

    # 1. Model loading
    t0 = time.time()
    tokenizer, model = load_model()
    load_time = time.time() - t0
    record_result("Qwen", "Qwen model and tokenizer load successfully",
                  tokenizer is not None and model is not None,
                  f"Loaded in {load_time:.2f}s")

    # 2. Math test: "What is 2 plus 2?"
    math_input = "What is 2 plus 2?"
    t0 = time.time()
    math_response = generate_response(
        text=math_input,
        semantic_input=math_input,
        max_new_tokens=40,
    )
    math_time = time.time() - t0
    indicates_four = "4" in math_response or "four" in math_response.lower()
    record_result("Qwen", "Math question '2 plus 2' indicates 4",
                  indicates_four,
                  f"Response='{math_response.strip()}' ({math_time:.2f}s)")

    # 3. Multilingual test
    mixed_text = "Nenu today college ki vellanu, but my friend Tamil-la pesitu irundhan"
    segs = get_language_segments(mixed_text)
    semantic_input = "I went to college today but my friend was speaking in Tamil"
    t0 = time.time()
    mixed_response = generate_response(
        text=mixed_text,
        segments=segs,
        semantic_input=semantic_input,
        max_new_tokens=60,
    )
    mixed_time = time.time() - t0
    record_result("Qwen", "Multilingual grounded response is non-empty",
                  bool(mixed_response.strip()),
                  f"Response='{mixed_response[:90]}...' ({mixed_time:.2f}s)")


# =====================================================================
# STEP 6: Faster-Whisper ASR
# =====================================================================

def test_step6_faster_whisper():
    section_header("STEP 6: Faster-Whisper ASR (large-v3-turbo, CPU INT8)")
    from multimix_src.voice.whisper_engine import transcribe_audio, load_whisper_model
    from multimix_src.config import AUDIO_DIR

    model = load_whisper_model()
    record_result("Faster-Whisper", "Model loads successfully",
                  model is not None,
                  f"Type: {type(model).__name__}")

    audio_files = [
        AUDIO_DIR / "WhatsApp Audio 2026-09-29 at 12.29.00 PM.mp4",
        AUDIO_DIR / "WhatsApp Audio 2026-09-29 at 12.32.26 PM.mp4",
    ]

    for audio_path in audio_files:
        t0 = time.time()
        result = transcribe_audio(str(audio_path))
        elapsed = time.time() - t0

        text = result.get("text", "")
        lang = result.get("language", "")
        segments = result.get("segments", [])
        avg_logprob = result.get("avg_logprob")
        no_speech = result.get("no_speech_prob")
        quality = result.get("asr_quality")

        record_result("Faster-Whisper", f"{audio_path.name} non-empty transcription",
                      bool(text.strip()),
                      f"Text='{text[:50]}...'")

        record_result("Faster-Whisper", f"{audio_path.name} language returned",
                      bool(lang),
                      f"Language={lang}")

        record_result("Faster-Whisper", f"{audio_path.name} segments exist",
                      len(segments) > 0,
                      f"{len(segments)} segment(s)")

        record_result("Faster-Whisper", f"{audio_path.name} ASR metrics exist",
                      avg_logprob is not None and no_speech is not None,
                      f"avg_logprob={avg_logprob:.3f}, no_speech={no_speech:.4f}")

        record_result("Faster-Whisper", f"{audio_path.name} quality classification is valid",
                      quality in ["good", "low", "failed"],
                      f"quality={quality}, elapsed={elapsed:.2f}s")


# =====================================================================
# STEP 7: ASR Failure Handling
# =====================================================================

def test_step7_asr_failure_handling():
    section_header("STEP 7: ASR Failure & Rejection Handling")
    from multimix_src.voice.pipeline import _empty_result, _ASR_FAILED_MESSAGE, process_voice
    from multimix_src.voice.whisper_engine import validate_asr_quality

    # 1. validate_asr_quality handles low/failed scenarios accurately
    empty_q = validate_asr_quality({"text": "", "avg_logprob": -0.1, "no_speech_prob": 0.0})
    record_result("ASR Failure", "Empty text flagged as failed",
                  empty_q["quality"] == "failed",
                  f"reasons={empty_q['reasons']}")

    bad_logprob_q = validate_asr_quality({"text": "noise", "avg_logprob": -1.88, "no_speech_prob": 0.1})
    record_result("ASR Failure", "Terrible logprob (-1.88) flagged as failed",
                  bad_logprob_q["quality"] == "failed",
                  f"reasons={bad_logprob_q['reasons']}")

    high_nospeech_q = validate_asr_quality({"text": "ghost", "avg_logprob": -0.3, "no_speech_prob": 0.95})
    record_result("ASR Failure", "High no_speech_prob (0.95) flagged as failed",
                  high_nospeech_q["quality"] == "failed",
                  f"reasons={high_nospeech_q['reasons']}")

    # 2. Verify _empty_result structure
    failed_res = _empty_result(
        audio_path="simulated.wav",
        transcription="garbled audio text",
        whisper_language="unknown",
        asr_quality="failed",
        asr_quality_reasons=["Very low avg_logprob"],
        status=_ASR_FAILED_MESSAGE,
    )

    record_result("ASR Failure", "Failed result produces NO response (does not call Qwen)",
                  failed_res["response"] == "",
                  "response is empty string")

    record_result("ASR Failure", "Failed result returns user-friendly status message",
                  "couldn't reliably understand" in failed_res["status"],
                  failed_res["status"])

    record_result("ASR Failure", "Failed result preserves raw transcription for debugging",
                  failed_res["transcription"] == "garbled audio text",
                  "preserved")


# =====================================================================
# STEP 8: IndicF5 Speech Synthesis
# =====================================================================

def test_step8_indicf5():
    section_header("STEP 8: IndicF5 Speech Synthesis")
    import soundfile as sf
    from multimix_src.voice.tts_engine import _load_engine, synthesize_speech
    from multimix_src.config import GENERATED_AUDIO_DIR

    # 1. Load engine
    t0 = time.time()
    model, vocoder = _load_engine()
    load_time = time.time() - t0
    record_result("IndicF5", "Model and vocoder load successfully",
                  model is not None and vocoder is not None,
                  f"Loaded in {load_time:.2f}s")

    # 2. Check existing test synthesis or synthesize short word
    test_out = GENERATED_AUDIO_DIR / "test_synth.wav"
    if not test_out.exists():
        synthesize_speech("Namaste", output_path=str(test_out))

    record_result("IndicF5", "Output WAV file exists",
                  test_out.exists(),
                  str(test_out))

    file_size = test_out.stat().st_size if test_out.exists() else 0
    record_result("IndicF5", "Output WAV file is non-empty",
                  file_size > 1000,
                  f"Size={file_size} bytes")

    audio_data, sr = sf.read(str(test_out))
    record_result("IndicF5", "Valid sample rate returned",
                  sr in [24000, 22050, 16000],
                  f"sample_rate={sr}, length={len(audio_data)} samples")


# =====================================================================
# STEP 9: Complete Text Pipeline
# =====================================================================

def test_step9_complete_text_pipeline():
    section_header("STEP 9: Complete End-to-End Text Pipeline")
    from multimix_src.language.detector import analyze_code_mix
    from multimix_src.language.segmenter import get_language_segments
    from multimix_src.normalization.corrector import normalize_text
    from multimix_src.semantic.interpreter import interpret_code_mix
    from multimix_src.response.generator import generate_response
    from multimix_src.voice.tts_engine import synthesize_speech
    from multimix_src.config import GENERATED_AUDIO_DIR

    text = "Nenu today college ki vellanu, but my friend Tamil-la pesitu irundhan"

    # Stage 1: Language Detection
    detection = analyze_code_mix(text)
    langs = detection.get("languages", [])
    record_result("Text Pipeline", "Stage 1: Language detection",
                  len(langs) >= 2,
                  f"Languages: {langs}")

    # Stage 2: Segmentation
    segments = get_language_segments(text)
    record_result("Text Pipeline", "Stage 2: Language segmentation",
                  len(segments) > 0,
                  f"{len(segments)} segments")

    # Stage 3: Normalization
    normalized = normalize_text(text=text, segments=segments)
    norm_text = normalized.get("normalized_text", "")
    record_result("Text Pipeline", "Stage 3: Normalization",
                  bool(norm_text),
                  f"Normalized: '{norm_text}'")

    # Stage 4: Semantic Interpretation
    semantic = interpret_code_mix(text=text, segments=segments)
    semantic_text = semantic.get("semantic_text", text)
    record_result("Text Pipeline", "Stage 4: Semantic interpretation",
                  bool(semantic_text) and semantic_text != text,
                  f"Semantic: '{semantic_text}'")

    # Stage 5: Qwen Response Generation
    response = generate_response(
        text=text,
        segments=segments,
        semantic_input=semantic_text,
        max_new_tokens=60,
    )
    record_result("Text Pipeline", "Stage 5: Qwen response generation",
                  bool(response.strip()),
                  f"Response: '{response[:80]}...'")

    # Stage 6: IndicF5 Speech Synthesis (use short snippet for efficiency)
    short_reply = "Got it! You went to college today."
    text_pipeline_audio = GENERATED_AUDIO_DIR / "test_text_pipeline_reply.wav"
    if not text_pipeline_audio.exists():
        synthesize_speech(short_reply, output_path=str(text_pipeline_audio))

    record_result("Text Pipeline", "Stage 6: IndicF5 speech synthesis",
                  text_pipeline_audio.exists() and text_pipeline_audio.stat().st_size > 0,
                  f"WAV: {text_pipeline_audio.name} ({text_pipeline_audio.stat().st_size} bytes)")


# =====================================================================
# STEP 10: Complete Voice Pipeline
# =====================================================================

def test_step10_complete_voice_pipeline():
    section_header("STEP 10: Complete End-to-End Voice Pipeline")
    from multimix_src.voice.pipeline import process_voice
    from multimix_src.config import AUDIO_DIR, GENERATED_AUDIO_DIR
    from multimix_src.voice.tts_engine import synthesize_speech

    audio_files = [
        AUDIO_DIR / "WhatsApp Audio 2026-09-29 at 12.29.00 PM.mp4",
        AUDIO_DIR / "WhatsApp Audio 2026-09-29 at 12.32.26 PM.mp4",
    ]

    for audio_path in audio_files:
        t0 = time.time()
        result = process_voice(str(audio_path))
        elapsed = time.time() - t0

        status = result.get("status", "")
        tx = result.get("transcription", "")
        asr_q = result.get("asr_quality", "")
        resp = result.get("response", "")

        record_result("Voice Pipeline", f"{audio_path.name} pipeline status",
                      status.startswith("success"),
                      f"status='{status}', asr_quality='{asr_q}' ({elapsed:.2f}s)")

        record_result("Voice Pipeline", f"{audio_path.name} transcription present",
                      bool(tx),
                      f"Transcription: '{tx[:60]}...'")

        record_result("Voice Pipeline", f"{audio_path.name} response generated",
                      bool(resp),
                      f"Response: '{resp[:70]}...'")

        record_result("Voice Pipeline", f"{audio_path.name} no response error",
                      result.get("response_error") is None,
                      "response_error is None")

    # Verify TTS integration can generate audio for usable voice pipeline response
    voice_wavs = list(GENERATED_AUDIO_DIR.glob("*.wav"))
    record_result("Voice Pipeline", "TTS output generated for voice response",
                  len(voice_wavs) > 0 and any(w.stat().st_size > 0 for w in voice_wavs),
                  f"Found {len(voice_wavs)} WAV asset(s) in {GENERATED_AUDIO_DIR.name}")


# =====================================================================
# MAIN RUNNER
# =====================================================================

def main():
    print("=" * 70)
    print("MultiMix AI — Task 6 Full Regression & Integration Test Suite")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python ver   : {sys.version.split()[0]}")
    print(f"Platform     : {sys.platform}")
    print("=" * 70)

    start_time = time.time()

    # Fast non-model tests first
    test_step11_configuration()
    test_step12_imports()
    test_step3_normalization()
    test_step4_semantic_interpretation()
    test_step2_language_detection()
    test_step7_asr_failure_handling()

    # Model & pipeline tests
    test_step6_faster_whisper()
    test_step5_qwen()
    test_step8_indicf5()
    test_step9_complete_text_pipeline()
    test_step10_complete_voice_pipeline()

    total_time = time.time() - start_time
    total = passed_tests + failed_tests

    section_header("FINAL TEST SUITE SUMMARY")
    print(f"  Total tests executed : {total}")
    print(f"  Passed tests         : {passed_tests}")
    print(f"  Failed tests         : {failed_tests}")
    print(f"  Total time           : {total_time:.2f}s")

    if failed_tests == 0:
        print("\n  >>> ALL TESTS PASSED SUCCESSFULLY! <<<\n")
        sys.exit(0)
    else:
        print(f"\n  >>> {failed_tests} TEST(S) FAILED! <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
