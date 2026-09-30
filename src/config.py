import os
from pathlib import Path

# ============================================================
# Dynamic Project Root Resolution
# ============================================================
# Priority:
# 1. MULTIMIX_PROJECT_ROOT environment variable (if set and non-empty)
# 2. Calculated dynamically from config file position (parent of src/)
# ============================================================

_env_root = os.environ.get("MULTIMIX_PROJECT_ROOT")
if _env_root and _env_root.strip():
    PROJECT_ROOT = Path(_env_root.strip()).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Core Directories
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"
GENERATED_AUDIO_DIR = AUDIO_DIR / "generated"
UPLOADS_AUDIO_DIR = AUDIO_DIR / "uploads"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Model Paths
QWEN_MODEL_PATH = MODELS_DIR / "qwen2.5-1.5b-instruct"
INDICF5_MODEL_DIR = MODELS_DIR / "indicf5"
WHISPER_MODEL_DIR = MODELS_DIR / "whisper"
WHISPER_TURBO_MODEL_DIR = MODELS_DIR / "whisper-turbo"
INDICLID_MODEL_PATH = MODELS_DIR / "indic_lid" / "model_baseline_roman.bin"
MT5_MODEL_DIR = MODELS_DIR / "mt5-small"

# Ensure required runtime output directories exist
GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
