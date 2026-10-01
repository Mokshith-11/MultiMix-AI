import sys
from pathlib import Path

from multimix_src.config import INDICF5_MODEL_DIR, AUDIO_DIR

ROOT = INDICF5_MODEL_DIR
sys.path.insert(0, str(ROOT))

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_vocoder, load_model, infer_process
import soundfile as sf


MODEL_PATH = ROOT / "model.safetensors"
VOCAB_PATH = ROOT / "checkpoints" / "vocab.txt"

REFERENCE_AUDIO = ROOT / "prompts" / "PAN_F_HAPPY_00001.wav"
REFERENCE_TEXT = "Namaste, aaj aap kaise hain?"

OUTPUT_PATH = AUDIO_DIR / "indicf5_test.wav"

MODEL_CONFIG = {
    "dim": 1024,
    "depth": 22,
    "heads": 16,
    "ff_mult": 2,
    "text_dim": 512,
    "conv_layers": 4,
}


print("=" * 60)
print("MultiMix AI - IndicF5 Standalone Test")
print("=" * 60)

print("\n[1/4] Checking files...")

for path in [MODEL_PATH, VOCAB_PATH, REFERENCE_AUDIO]:
    print(f"{path.name}: {'OK' if path.exists() else 'MISSING'}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not VOCAB_PATH.exists():
    raise FileNotFoundError(VOCAB_PATH)

if not REFERENCE_AUDIO.exists():
    raise FileNotFoundError(REFERENCE_AUDIO)


print("\n[2/4] Loading vocoder...")
vocoder = load_vocoder()
print("Vocoder loaded.")


print("\n[3/4] Loading IndicF5 model...")
model = load_model(
    DiT,
    MODEL_CONFIG,
    str(MODEL_PATH),
    vocab_file=str(VOCAB_PATH),
)

print("IndicF5 model loaded.")


print("\n[4/4] Generating speech...")

TEXT = "నేను ఈ రోజు కాలేజీకి వెళ్లాను."

final_wave, sample_rate, spectrogram = infer_process(
    str(REFERENCE_AUDIO),
    REFERENCE_TEXT,
    TEXT,
    model,
    vocoder,
    cross_fade_duration=0.15,
    speed=1.0,
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

sf.write(
    str(OUTPUT_PATH),
    final_wave,
    sample_rate,
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)
print(f"Output : {OUTPUT_PATH}")
print(f"Sample rate : {sample_rate}")
print(f"Audio samples : {len(final_wave)}")
print("=" * 60)
