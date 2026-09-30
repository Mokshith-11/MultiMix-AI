# MultiMix AI — Testing Strategy & Verification Suites

This document provides a comprehensive guide to the automated test suites, testing methodology, and regression verification results for MultiMix AI.

---

## 1. Testing Philosophy & Scope

The MultiMix AI test framework prioritizes **software regression and pipeline integration testing**. 

### What the Test Suite Validates
- **Pipeline Contracts:** Ensures inputs and outputs between adjacent subsystems adhere strictly to expected schemas.
- **Circuit Breakers & Graceful Degradation:** Confirms that when audio quality is low or ASR fails, the system intercepts the error before reaching the LLM or TTS, returning user-friendly messages without application crashes.
- **Portability & Paths:** Confirms that directory and model paths resolve dynamically across different operating systems without hardcoded absolute paths.
- **Grounding & NLP Logic:** Tests token-level segmentation, slang normalization, and semantic reconstruction across Indian code-mixed examples.

### What the Test Suite Does NOT Represent
- **Statistical Model Accuracy:** Passing these integration tests confirms that models load and execute according to pipeline contracts; it **does not** imply 100% Word Error Rate (WER) accuracy or perfect transcription of all regional dialects.

---

## 2. Test Suites Overview

| Test Suite File | Focus Area | Test Count |
| :--- | :--- | :---: |
| [`tests/test_task2_regression.py`](../tests/test_task2_regression.py) | Faster-Whisper upgrade validation, standalone ASR, text pipeline, voice pipeline | 3 Test Suites |
| [`tests/test_task3_regression.py`](../tests/test_task3_regression.py) | ASR quality validation gate, synthetic quality scoring, circuit breaker handling | 29 Checks |
| [`tests/test_task6_regression.py`](../tests/test_task6_regression.py) | Comprehensive end-to-end regression across all 11 functional subsystems | 91 Checks |
| **Streamlit Sanity Checks** | Compilation (`py_compile`) and headless web app startup verification | 2 Checks |
| **Cumulative Total** | **All Verification Checks** | **122 / 122 Passed** |

---

## 3. Detailed Test Phases

### Phase 1: Task 2 Faster-Whisper Regression (`test_task2_regression.py`)
- Evaluated `faster-whisper large-v3-turbo` against both real-world WhatsApp audio recordings.
- Tested:
  1. `transcribe_audio()` direct invocation on CPU INT8.
  2. Text pipeline isolation (`segmenter → normalizer → semantic → Qwen`).
  3. Integrated voice pipeline execution (`process_voice()`).

### Phase 2: Task 3 ASR Quality Validation Gate (`test_task3_regression.py`)
- Validated the pre-LLM quality validation gate:
  1. Standalone unit tests for `validate_asr_quality()` covering good audio, marginal logprobs, elevated no-speech probabilities, repetitive audio, empty transcripts, and corrupted transcripts.
  2. Failure handling test ensuring uninterpretable audio halts processing and provides actionable guidance to the user.
  3. Regression checks confirming text pipeline remains unimpacted by voice quality checks.

### Phase 3: Task 6 Full System Regression (`test_task6_regression.py`)
A unified test suite validating the entire application across 11 core categories:
1. **Dynamic Configuration:** Dynamic resolution of project roots and model paths (10 checks).
2. **Core Imports:** Clean import verification for all 10 core Python modules (10 checks).
3. **Language Detection:** Native scripts and Romanized lexicons across Telugu, Tamil, Hindi, Bengali, English, and a 3-language code-mix (12 checks).
4. **Normalization:** Abbreviation expansions and multilingual token preservation (14 checks).
5. **Semantic Interpretation:** 5 bilingual pairs and 3-language code-mixed semantic reconstruction (6 checks).
6. **Local Qwen Response Generation:** Offline model loading, basic math reasoning, and multilingual grounded responses (3 checks).
7. **Faster-Whisper Standalone:** Real audio transcription, language detection, and segment diagnostic metrics (11 checks).
8. **ASR Quality Rejection:** Boundary testing and graceful failure response verification (7 checks).
9. **IndicF5 Speech Synthesis:** Model/vocoder loading, WAV file emission, and sample rate checks (4 checks).
10. **Complete Text Pipeline:** End-to-end execution from input text to synthetic audio response (6 checks).
11. **Complete Voice Pipeline:** End-to-end execution from WhatsApp audio to synthetic voice response (8 checks).

---

## 4. How to Execute Tests

Activate the virtual environment and run the test scripts:

### Execute Task 6 Comprehensive Suite
```bash
python tests/test_task6_regression.py
```

### Execute Task 3 Quality Gate Suite
```bash
python tests/test_task3_regression.py
```

### Execute Task 2 Faster-Whisper Suite
```bash
python tests/test_task2_regression.py
```

### Run Streamlit Integrity Check
```bash
python -m py_compile app.py
```

---

## 5. Summary of Results

All 122 automated checks passed with 0 failures:
- Core configuration: **10/10 PASS**
- Module imports: **10/10 PASS**
- Language detection: **12/12 PASS**
- Normalization: **14/14 PASS**
- Semantic interpretation: **6/6 PASS**
- Qwen generation: **3/3 PASS**
- Faster-Whisper ASR: **11/11 PASS**
- Quality validation & rejection: **7/7 PASS**
- IndicF5 TTS: **4/4 PASS**
- Complete text pipeline: **6/6 PASS**
- Complete voice pipeline: **8/8 PASS**
- Streamlit integrity: **2/2 PASS**
- Baseline Task 3 regression: **29/29 PASS**

Full execution logs and breakdown details can be reviewed in [`docs/TASK6_TEST_REPORT.md`](TASK6_TEST_REPORT.md).
