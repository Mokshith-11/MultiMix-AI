import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATED_AUDIO_DIR = PROJECT_ROOT / "outputs"
GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
WHISPER_MODEL_ID = "large-v3-turbo"
INDICF5_MODEL_ID = "ai4bharat/IndicF5"

M2M100_MODEL_ID = "facebook/m2m100_418M"
TELUGU_MODEL_ID = "anithasoma/nllb-finetuned-telugu"

REFERENCE_TEXT = (
    "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ "
    "ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ ਹਨ।"
)

SUPPORTED_LANGUAGES = [
    "Telugu",
    "Tamil",
    "Hindi",
    "Bengali",
    "English",
]
