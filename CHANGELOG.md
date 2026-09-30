# MultiMix AI — Changelog

All notable technical milestones, feature additions, and architectural iterations for MultiMix AI are documented in this file.

---

## [1.0.0] - Full System Regression & Handover
### Completed
- **Task 6 Regression Testing:** Executed full integration test suite across all 11 functional subsystems (122/122 checks passed with 0 failures).
- **Bug Fixes:**
  - Added missing `Dict` import in `src/response/voice_response.py`.
  - Added safe standard UTF-8 console output guards in test runners for Windows compatibility.
- **Project Documentation:** Complete technical and client-facing documentation suite created (`README.md`, `ARCHITECTURE.md`, `INSTALLATION.md`, `TESTING.md`, `DEPLOYMENT.md`, `CLIENT_HANDOVER.md`).

---

## [0.9.0] - Portable Configuration & Workspace Normalization
### Added
- Dynamic project root and directory path resolution in `src/config.py`.
- Support for `MULTIMIX_PROJECT_ROOT` environment override.
- Automated creation of runtime audio directories (`assets/audio/generated`, `assets/audio/uploads`).
- Elimination of hardcoded drive paths across core modules.

---

## [0.8.0] - Streamlit UI Polish & Dual-Tab Interface
### Added
- Polished Streamlit web application (`app.py`).
- Two-tab layout: **Text Interaction** and **Voice Interaction**.
- ASR quality indicator banners (🟢 Good, 🟡 Low Confidence, 🔴 Unusable).
- Audio player widgets for user uploads and synthesized voice responses.
- Collapsible Developer Details expanders displaying confidence scores and segmentation details.

---

## [0.7.0] - ASR Quality Validation Gate & Circuit Breakers
### Added
- Diagnostic metric extraction in `whisper_engine.py` (`avg_logprob`, `no_speech_prob`, `compression_ratio`).
- Pre-LLM quality validation function `validate_asr_quality()` in `src/voice/whisper_engine.py`.
- Graceful error interception: uninterpretable audio halts processing and provides user guidance without invoking LLM or TTS.
- Task 3 regression test suite (`tests/test_task3_regression.py`) covering synthetic and real quality scenarios.

---

## [0.6.0] - Faster-Whisper large-v3-turbo Upgrade
### Added
- Upgraded primary ASR engine from Whisper Small to `faster-whisper large-v3-turbo` with INT8 quantization on CPU.
- Standalone diagnostic tool `tests/test_faster_whisper.py` comparing transcription quality.
- Task 2 regression test suite (`tests/test_task2_regression.py`).

---

## [0.5.0] - IndicF5 Speech Synthesis Integration
### Added
- Local neural TTS engine in `src/voice/tts_engine.py` utilizing the IndicF5 Diffusion Transformer (`DiT`).
- Integrated `Vocos` 24kHz neural vocoder for high-fidelity waveform generation.
- Audio synthesis pipeline producing standardized 24,000 Hz WAV responses.

---

## [0.4.0] - Local Qwen 2.5 LLM Integration
### Added
- Offline conversational response generator in `src/response/generator.py` powered by `Qwen2.5-1.5B-Instruct`.
- Grounded prompt construction incorporating structured semantic interpretations.
- Zero-API-key local inference on CPU with PyTorch.

---

## [0.3.0] - Semantic Interpretation & Grounding
### Added
- Rule-grounded semantic translator in `src/semantic/interpreter.py`.
- Bilingual dictionary mappings for Telugu, Tamil, Hindi, and Bengali Romanized words.
- Contextual clause conversion (e.g. `Tamil-la` → `in Tamil`).
- Pattern-based Subject-Object-Verb to Subject-Verb-Object reconstruction.

---

## [0.2.0] - Text Normalization & Slang Correction
### Added
- Text corrector and normalization engine in `src/normalization/corrector.py`.
- Expansion of informal SMS abbreviations (`tmrw` → `tomorrow`, `clg` → `college`, `frnd` → `friend`, `plz` → `please`, `bcoz` → `because`).
- Repeated character reduction (`todayyy` → `today`).
- Multilingual token preservation avoiding erroneous English autocorrection.

---

## [0.1.0] - Initial Multilingual Foundation
### Added
- Core project directory structure and basic configuration.
- Native Unicode script detection for Indic languages in `src/language/detector.py`.
- Romanized lexicon matching for Telugu, Tamil, Hindi, and Bengali.
- IndicLID fastText integration for Romanized language identification.
- Token-level language segmentation with confidence scoring in `src/language/segmenter.py`.
- Contextual ambiguity resolution for short particles (`ki`, `lo`, `la`, `me`).
