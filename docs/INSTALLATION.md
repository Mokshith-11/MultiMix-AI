# MultiMix AI — Installation & Setup Guide

This guide details the steps required to set up and run MultiMix AI in a local environment.

---

## 1. Prerequisites

### Hardware Requirements
- **Processor:** Modern x86_64 multi-core CPU (Intel Core i5/i7/i9 or AMD Ryzen 5/7/9).
- **RAM:** Minimum 16 GB system memory (32 GB recommended when running all models locally on CPU).
- **Disk Space:** At least 20 GB of free disk space for models, virtual environment, and runtime assets.
- *(Optional GPU):* NVIDIA GPU with 8GB+ VRAM and CUDA support can significantly accelerate inference if PyTorch CUDA is installed.

### Software Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 22.04+, or macOS (x86_64 / Apple Silicon).
- **Python:** Python 3.12 (64-bit).
- **Git:** Standard Git version control.

---

## 2. Environment Setup

### 2.1. Clone Repository
```bash
git clone <repository-url>
cd MultiMix-AI
```

### 2.2. Create Virtual Environment
Create a clean virtual environment using Python 3.12:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2.3. Install Dependencies
Upgrade pip and install requirements:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Model Directory Structure & Placement

MultiMix AI expects models to be placed in the `models/` directory at the project root. This directory is excluded from Git tracking via `.gitignore`.

Verify or populate the `models/` folder as follows:

```
MultiMix-AI/models/
├── qwen2.5-1.5b-instruct/
│   ├── config.json
│   ├── model.safetensors (or model.safetensors.index.json)
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── whisper-turbo/
│   ├── model.bin (or model.safetensors)
│   ├── config.json
│   ├── tokenizer.json
│   └── vocabulary.json
├── indicf5/
│   ├── model.safetensors
│   ├── checkpoints/
│   │   └── vocab.txt
│   └── prompts/
│       └── PAN_F_HAPPY_00001.wav
├── indic_lid/
│   └── model_baseline_roman.bin
└── mt5-small/
    ├── config.json
    └── pytorch_model.bin (or model.safetensors)
```

### Model Sources

1. **Qwen 2.5 1.5B Instruct:**
   - Hugging Face identifier: `Qwen/Qwen2.5-1.5B-Instruct`
2. **faster-whisper large-v3-turbo:**
   - Downloaded automatically via `faster-whisper` library or downloaded directly from Hugging Face: `Systran/faster-whisper-large-v3-turbo`
3. **IndicF5:**
   - Checkpoints from AI4Bharat IndicF5 release, placed in `models/indicf5`
4. **IndicLID:**
   - AI4Bharat Romanized LID fastText model: `model_baseline_roman.bin` placed in `models/indic_lid/`
5. **Vocos Vocoder:**
   - Downloaded automatically by the `vocos` library to Hugging Face cache (`charactr/vocos-mel-24khz`).

---

## 4. Configuration

All path resolution is handled dynamically in `src/config.py`.

### Environment Variables (Optional)
If you wish to place the MultiMix AI root directory in a custom location, you can set the `MULTIMIX_PROJECT_ROOT` environment variable:

```powershell
$env:MULTIMIX_PROJECT_ROOT = "C:\path\to\MultiMix-AI"
```

If not set, the project root is automatically determined from the relative position of `src/config.py`.

No API keys are required.

---

## 5. Running the Application

Activate your virtual environment and launch Streamlit:

```bash
streamlit run app.py
```

Streamlit will print the local server URL (e.g., `http://localhost:8501`). Open this address in your web browser.

---

## 6. CPU Performance Expectations

Because MultiMix AI operates locally without cloud API acceleration, execution times on CPU are as follows:

| Stage | Model / Component | Typical Latency on Modern CPU |
| :--- | :--- | :--- |
| **Language Detection & Segmentation** | Rule lexicons + fastText | < 0.1 seconds |
| **Semantic Interpretation** | Rule-based engine | < 0.05 seconds |
| **Speech-to-Text (ASR)** | faster-whisper large-v3-turbo (INT8) | ~50–100 seconds per 5–10s audio clip |
| **Response Generation (LLM)** | Qwen 2.5 1.5B Instruct | ~15–30 seconds (for ~50 tokens) |
| **Speech Synthesis (TTS)** | IndicF5 DiT ODE Solver | ~2–3 minutes per sentence |

*Note: For production deployments requiring sub-second response times, running on an NVIDIA GPU (CUDA) is strongly advised.*

---

## 7. Troubleshooting

### 1. `ModuleNotFoundError: No module named 'torch'`
Ensure that your virtual environment is active before running commands (`.\.venv\Scripts\activate` or `source .venv/bin/activate`). Verify using `python -c "import torch; print(torch.__file__)"`.

### 2. Windows Unicode Encoding Errors in Console
If running custom scripts in a Windows command prompt that outputs non-ASCII characters, set the environment variable:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### 3. Missing Model Checkpoints
If a model path error occurs, verify that the files listed in Section 3 are present in your local `models/` directory.

### 4. Excessive Streamlit Reloads
If Streamlit re-triggers unnecessarily when audio files are generated, run Streamlit with file-watching disabled:
```bash
streamlit run app.py --server.fileWatcherType none
```
