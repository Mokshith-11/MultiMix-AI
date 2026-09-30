"""
Diagnostic: faster-whisper large-v3-turbo vs existing Whisper small
Tests both audio files and prints quality metrics.
"""
import time
from pathlib import Path
from faster_whisper import WhisperModel

from src.config import WHISPER_TURBO_MODEL_DIR, AUDIO_DIR

MODEL_ROOT = WHISPER_TURBO_MODEL_DIR
AUDIO_DIR = AUDIO_DIR

AUDIO_FILES = [
    AUDIO_DIR / "WhatsApp Audio 2026-09-29 at 12.29.00 PM.mp4",
    AUDIO_DIR / "WhatsApp Audio 2026-09-29 at 12.32.26 PM.mp4",
]

def main():
    print("=" * 60)
    print("FASTER-WHISPER large-v3-turbo DIAGNOSTIC")
    print("=" * 60)

    print("\nLoading model (this may take a moment on CPU)...")
    load_start = time.time()
    model = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        download_root=str(MODEL_ROOT),
    )
    print(f"Model loaded in {time.time() - load_start:.1f}s\n")

    for audio_path in AUDIO_FILES:
        print("-" * 60)
        print(f"FILE: {audio_path.name}")
        print(f"SIZE: {audio_path.stat().st_size} bytes")
        print("-" * 60)

        start = time.time()
        segments_gen, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
        )

        # Consume the generator and collect segments
        segments = list(segments_gen)
        elapsed = time.time() - start

        # Language info from TranscriptionInfo
        print(f"Detected language   : {info.language}")
        print(f"Language probability : {info.language_probability:.4f}")
        print(f"Duration (audio)    : {info.duration:.2f}s")
        print(f"Processing time     : {elapsed:.2f}s")
        print(f"Realtime factor     : {elapsed / max(info.duration, 0.01):.2f}x")
        print()

        # Full transcription
        full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        print(f"TRANSCRIPTION: {full_text}")
        print()

        # Per-segment details
        print("SEGMENTS:")
        for i, seg in enumerate(segments):
            print(
                f"  [{i}] {seg.start:.2f}s - {seg.end:.2f}s | "
                f"avg_logprob={seg.avg_logprob:.4f} | "
                f"no_speech_prob={seg.no_speech_prob:.4f} | "
                f"compression_ratio={seg.compression_ratio:.4f}"
            )
            print(f"       TEXT: {seg.text.strip()}")

        # Aggregate quality metrics
        if segments:
            avg_lp = sum(s.avg_logprob for s in segments) / len(segments)
            avg_nsp = sum(s.no_speech_prob for s in segments) / len(segments)
            avg_cr = sum(s.compression_ratio for s in segments) / len(segments)
            print(f"\n  AGGREGATE avg_logprob       : {avg_lp:.4f}")
            print(f"  AGGREGATE no_speech_prob    : {avg_nsp:.4f}")
            print(f"  AGGREGATE compression_ratio : {avg_cr:.4f}")
        else:
            print("\n  NO SEGMENTS PRODUCED")

        print("\n")

    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
