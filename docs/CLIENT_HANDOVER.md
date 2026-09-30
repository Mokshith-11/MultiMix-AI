# MultiMix AI — Client Handover & Operational Manual

**Project Name:** MultiMix AI  
**Handover Version:** 1.0.0-MVP  
**Document Classification:** Client & Technical Handover  

---

## 1. Project Overview & Purpose

MultiMix AI is a localized conversational artificial intelligence system built to understand and engage in natural, code-mixed Indian-language conversations. In everyday Indian discourse, speakers routinely interleave regional languages with English within the same sentence (e.g. Telugu-English, Tamil-English, Hindi-English).

MultiMix AI processes these mixed inputs across both **text** and **voice** modalities, performing token-level segmentation, slang normalization, and semantic grounding before generating helpful conversational replies in both text and synthesized speech.

---

## 2. Supported Languages

| Language | Romanized (Latin Script) Support | Native Script Support | Example Code-Mixed Constituent |
| :--- | :---: | :---: | :--- |
| **Telugu** | Yes | Yes (తెలుగు) | `Nenu today college ki vellanu` |
| **Tamil** | Yes | Yes (தமிழ்) | `my friend Tamil-la pesitu irundhan` |
| **Hindi** | Yes | Yes (हिन्दी / Devanagari) | `Main aaj college gaya` |
| **Bengali** | Yes | Yes (বাংলা) | `Ami aaj college jabo` |
| **English** | Yes | Yes (Latin) | Base vocabulary across all mixed inputs |

---

## 3. Implemented Features Summary

### Text Pipeline
- Hybrid language identification combining script Unicode analysis, Romanized lexicons, and fastText embeddings.
- Token-level language segmentation with confidence indicators.
- Chat/SMS abbreviation normalization (`tmrw` → `tomorrow`, `clg` → `college`, `plz` → `please`).
- Contextual resolution of short ambiguous particles (`ki`, `lo`, `la`, `me`).
- Semantic grounding that translates code-mixed sentences into structured English meanings.
- Local conversational response generation using Qwen 2.5 1.5B Instruct.
- Optional voice reply generation using IndicF5 neural TTS.

### Voice Pipeline
- Speech-to-text transcription powered by `faster-whisper large-v3-turbo` (INT8 quantization).
- Pre-LLM ASR quality validation gate that assesses average log-probability, silence probability, and compression ratio.
- Graceful circuit breaker: Low-confidence or uninterpretable speech is caught immediately, prompting the user for clearer input without crashing.
- Automatic handoff of valid transcriptions to the downstream multilingual NLP engine.
- Synthetic voice reply generation using the IndicF5 Diffusion Transformer and Vocos 24kHz vocoder.

### User Interface
- Streamlit web interface with distinct **Text Interaction** and **Voice Interaction** tabs.
- Clear audio playback widgets for uploaded and generated audio.
- Collapsible Developer Details expanders for technical inspection and confidence metrics.

---

## 4. Model & Component Stack

| Subsystem | Model / Component | Configuration | Purpose |
| :--- | :--- | :--- | :--- |
| **ASR** | `faster-whisper large-v3-turbo` | CPU / INT8 Quantization | Fast multilingual speech-to-text |
| **LLM** | `Qwen2.5-1.5B-Instruct` | CPU / Float32 | Local conversational response generation |
| **TTS** | `IndicF5` (DiT Architecture) | CPU / Float32 | Neural speech synthesis |
| **Vocoder** | `Vocos` (charactr/vocos-mel-24khz) | CPU / Float32 | 24kHz audio waveform generation |
| **LID** | `IndicLID` (AI4Bharat) | fastText Romanized | Supporting Romanized language ID |

*Note: All models run 100% locally. Zero external API calls are made, and zero third-party API keys are required.*

---

## 5. Quick-Start Operational Instructions

### Prerequisites
- Python 3.12 (64-bit)
- 16 GB+ RAM

### Starting the Application
From the project root directory, activate the virtual environment and launch Streamlit:

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

```bash
# Linux / macOS
source .venv/bin/activate
streamlit run app.py
```

Open the displayed address (e.g. `http://localhost:8501`) in any modern web browser.

---

## 6. Verification & Test Status

The system has undergone full regression and integration testing under **Task 6**:
- **Task 6 Comprehensive Suite:** 91 / 91 passed
- **Task 3 Quality Gate Baseline:** 29 / 29 passed
- **Streamlit Compilation & Startup:** 2 / 2 passed
- **Total Validated Checks:** **122 / 122 (100% pass rate, 0 failures)**

To rerun verification tests at any time:
```bash
python tests/test_task6_regression.py
```

---

## 7. Known Limitations

1. **CPU Latency:**
   - On CPU, speech recognition takes ~50–100 seconds per audio recording, and IndicF5 speech synthesis takes ~2–3 minutes per sentence. Real-time conversational speeds require deployment on an NVIDIA GPU (CUDA).
2. **ASR Script Representation:**
   - Faster-Whisper transcribes real-world code-mixed Indian speech using phonetic Devanagari or regional scripts. Downstream components normalize and interpret this correctly, but transcripts appear in phonetic script rather than Romanized Latin text.
3. **Hardware Storage:**
   - Local model weights require approximately 15 GB of disk space.

---

## 8. Recommended Future Improvements

1. **GPU Acceleration:** Deploying on an NVIDIA A10G or L4 GPU will reduce end-to-end response latency from minutes to under 3 seconds.
2. **Streaming ASR & Audio:** Incorporating WebSocket-based chunked streaming for real-time speech transcription and audio output streaming.
3. **Fine-Tuning Indic ASR:** Fine-tuning Whisper specifically on colloquial Romanized transliterations to output Romanized English text directly.
4. **Direct Microphone Recording:** Adding client-side JavaScript WebRTC/MediaRecorder for one-click browser voice recording.

---

## 9. Handover Checklist

- [x] Application source code (`app.py`, `src/`)
- [x] Automated test suite (`tests/`)
- [x] Requirements specification (`requirements.txt`)
- [x] Git exclusion rules (`.gitignore`)
- [x] System documentation (`README.md`, `CHANGELOG.md`, `docs/`)
- [x] Test report (`docs/TASK6_TEST_REPORT.md`)
- [x] Reference audio demo recordings (`assets/audio/`)
- [x] Verified zero API key dependencies
- [x] Verified 122/122 test pass status
