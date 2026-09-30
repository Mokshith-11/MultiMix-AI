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

# ============================================================
# Cloud & Model Availability Checks
# ============================================================

IS_STREAMLIT_CLOUD = bool(
    os.environ.get("STREAMLIT_RUNTIME")
    or os.environ.get("STREAMLIT_SHARING")
    or os.environ.get("IS_STREAMLIT_CLOUD")
)


def check_model_availability() -> dict:
    """
    Check the presence of local model files on disk.
    Does NOT load or download any models into memory.
    """
    indiclid_ok = bool(
        INDICLID_MODEL_PATH.exists()
        and INDICLID_MODEL_PATH.stat().st_size > 1000
    )

    qwen_ok = bool(
        QWEN_MODEL_PATH.exists()
        and (QWEN_MODEL_PATH / "config.json").exists()
        and (
            (QWEN_MODEL_PATH / "model.safetensors").exists()
            or (QWEN_MODEL_PATH / "model.safetensors.index.json").exists()
        )
    )

    whisper_ok = bool(
        WHISPER_TURBO_MODEL_DIR.exists()
        and (
            (WHISPER_TURBO_MODEL_DIR / "model.bin").exists()
            or bool(list(WHISPER_TURBO_MODEL_DIR.glob("models--*faster-whisper-large-v3-turbo*")))
            or (WHISPER_TURBO_MODEL_DIR / "config.json").exists()
        )
    )

    indicf5_ok = bool(
        INDICF5_MODEL_DIR.exists()
        and (INDICF5_MODEL_DIR / "model.safetensors").exists()
        and (INDICF5_MODEL_DIR / "checkpoints" / "vocab.txt").exists()
    )

    return {
        "indiclid": indiclid_ok,
        "qwen": qwen_ok,
        "whisper": whisper_ok,
        "indicf5": indicf5_ok,
    }


def get_deployment_mode() -> str:
    """
    Returns 'FULL LOCAL MODEL' if all models are present,
    or 'FREE CLOUD DEMO' if any model weight is unprovisioned.
    """
    avail = check_model_availability()
    if all(avail.values()):
        return "FULL LOCAL MODEL"
    return "FREE CLOUD DEMO"
