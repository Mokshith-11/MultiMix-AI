import base64
import os
import tempfile
import uuid
from pathlib import Path

import requests
import streamlit as st

try:
    from multimix_src.config import (
        GENERATED_AUDIO_DIR,
        UPLOADS_AUDIO_DIR,
        check_model_availability,
        get_deployment_mode,
    )
    from multimix_src.language.detector import analyze_code_mix
    from multimix_src.language.segmenter import get_language_segments
    from multimix_src.normalization.corrector import normalize_text
    from multimix_src.semantic.interpreter import interpret_code_mix
    from multimix_src.response.generator import generate_response
    from multimix_src.response.voice_response import generate_voice_response
    from multimix_src.voice.pipeline import process_voice
    from multimix_src.voice.tts_engine import synthesize_speech
except ImportError:
    from src.config import (
        GENERATED_AUDIO_DIR,
        UPLOADS_AUDIO_DIR,
        check_model_availability,
        get_deployment_mode,
    )
    from src.language.detector import analyze_code_mix
    from src.language.segmenter import get_language_segments
    from src.normalization.corrector import normalize_text
    from src.semantic.interpreter import interpret_code_mix
    from src.response.generator import generate_response
    from src.response.voice_response import generate_voice_response
    from src.voice.pipeline import process_voice
    from src.voice.tts_engine import synthesize_speech


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MultiMix AI",
    page_icon="🌐",
    layout="centered",
)

# Deployment and model availability status
model_status = check_model_availability()
deployment_mode = get_deployment_mode()

# Modal Backend Endpoint URL Resolution
DEFAULT_MODAL_PROCESS_URL = "https://vinnyvvinny8--multimix-ai-backend-process.modal.run"


def get_modal_backend_url() -> str:
    """Retrieve Modal process endpoint URL from env, secrets, or fallback default."""
    url = os.environ.get("MODAL_BACKEND_URL")
    if not url and hasattr(st, "secrets"):
        try:
            url = st.secrets.get("MODAL_BACKEND_URL")
        except Exception:
            pass
    if not url:
        url = DEFAULT_MODAL_PROCESS_URL
    return url.strip()


def call_modal_backend(payload: dict, timeout_seconds: int = 120) -> dict:
    """Send JSON payload over HTTPS to the deployed Modal GPU backend."""
    url = get_modal_backend_url()
    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error": "Modal GPU backend request timed out. Please try again.",
        }
    except requests.exceptions.RequestException as req_err:
        return {
            "status": "error",
            "error": f"Failed to connect to Modal GPU backend: {req_err}",
        }
    except Exception as err:
        return {
            "status": "error",
            "error": f"Unexpected error during Modal API call: {err}",
        }


# Language Display Mapping Helper
LANG_NAME_MAP = {
    "te": "Telugu", "telugu": "Telugu",
    "ta": "Tamil", "tamil": "Tamil",
    "hi": "Hindi", "hindi": "Hindi",
    "bn": "Bengali", "bengali": "Bengali",
    "en": "English", "english": "English",
}


def format_detected_languages(languages_list) -> str:
    """Format a list of languages nicely using bullet dots."""
    if not languages_list:
        return "None detected"
    formatted = []
    for lang in languages_list:
        clean = LANG_NAME_MAP.get(str(lang).lower().strip(), str(lang).title().strip())
        if clean not in formatted:
            formatted.append(clean)
    return " • ".join(formatted) if formatted else "None detected"


# ============================================================
# HEADER
# ============================================================

col_head, col_mode = st.columns([3, 1])
with col_head:
    st.title("MULTIMIX AI")
    st.subheader("Multilingual & Code-Mixed Conversational AI")
with col_mode:
    st.write("")
    if deployment_mode == "FULL LOCAL MODEL":
        st.success("🟢 **Full Local Model**")
    else:
        st.success("⚡ **Modal GPU Active**")

st.caption("Understand and respond to multilingual Indian code-mixed text and voice.")

if deployment_mode != "FULL LOCAL MODEL":
    st.info(
        "⚡ **Modal GPU Acceleration Active**: Text analysis, Qwen 2.5 1.5B LLM response generation, "
        "Whisper ASR, and IndicF5 TTS are powered by remote NVIDIA T4 GPU on Modal."
    )

st.divider()


# ============================================================
# NAVIGATION TABS
# ============================================================

tab_text, tab_voice = st.tabs(["💬 Text Interaction", "🎤 Voice Interaction"])


# ============================================================
# TAB 1: TEXT INTERACTION
# ============================================================

with tab_text:
    st.subheader("💬 Text Input")

    input_text = st.text_area(
        "Enter your multilingual / code-mixed message:",
        placeholder="Nenu today college ki vellanu, but my friend Tamil-la pesitu irundhan",
        height=120,
    )

    if st.button("Analyze & Respond", type="primary", use_container_width=True, key="btn_text_submit"):
        if not input_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            try:
                # ----------------------------------------------------
                # LOCAL MODEL EXECUTION (WHEN ALL WEIGHTS ARE LOCAL)
                # ----------------------------------------------------
                if deployment_mode == "FULL LOCAL MODEL":
                    with st.spinner("Analyzing multilingual input..."):
                        detection = analyze_code_mix(input_text)
                        segments = get_language_segments(input_text)
                        normalized = normalize_text(text=input_text, segments=segments)
                        semantic = interpret_code_mix(text=input_text, segments=segments)

                    det_langs = detection.get("languages", [])
                    if not det_langs and segments:
                        det_langs = list({s["language"] for s in segments if s.get("language")})

                    response_text = ""
                    if model_status.get("qwen"):
                        with st.spinner("Generating AI response..."):
                            response_text = generate_response(
                                text=input_text,
                                segments=segments,
                                semantic_input=semantic.get("semantic_text", input_text),
                                max_new_tokens=100,
                            )

                    audio_path = None
                    if response_text and model_status.get("indicf5"):
                        with st.spinner("Generating voice response..."):
                            try:
                                output_dir = GENERATED_AUDIO_DIR
                                out_file = output_dir / f"response_{uuid.uuid4().hex}.wav"
                                audio_path = synthesize_speech(
                                    text=response_text,
                                    output_path=str(out_file),
                                    speed=1.0,
                                )
                            except Exception as tts_err:
                                st.warning(f"Voice generation unavailable: {tts_err}")

                    norm_display_text = normalized.get("normalized_text", input_text)
                    sem_display_text = semantic.get("semantic_text", "")
                    audio_bytes = Path(audio_path).read_bytes() if (audio_path and Path(audio_path).exists()) else None
                    audio_mime = "audio/wav"

                # ----------------------------------------------------
                # MODAL GPU BACKEND EXECUTION (CLOUD / DEMO MODE)
                # ----------------------------------------------------
                else:
                    with st.spinner("Processing with Modal GPU Backend (Qwen 1.5B + IndicF5)..."):
                        payload = {
                            "mode": "text",
                            "text": input_text,
                            "target_language": "Auto",
                        }
                        modal_res = call_modal_backend(payload)

                    if modal_res.get("status") == "error":
                        st.error(f"Backend Error: {modal_res.get('error', 'Unknown backend error')}")
                        st.stop()

                    # Extract returned structures from Modal payload
                    response_text = modal_res.get("response", "")
                    segments = modal_res.get("segments", [])

                    norm_raw = modal_res.get("normalized", {})
                    if isinstance(norm_raw, dict):
                        norm_display_text = norm_raw.get("normalized_text", input_text)
                    else:
                        norm_display_text = str(norm_raw) if norm_raw else input_text

                    sem_raw = modal_res.get("semantic", {})
                    if isinstance(sem_raw, dict):
                        sem_display_text = sem_raw.get("semantic_text", "")
                        det_langs = sem_raw.get("languages", [])
                    else:
                        sem_display_text = str(sem_raw) if sem_raw else ""
                        det_langs = []

                    if not det_langs and segments:
                        det_langs = list({s["language"] for s in segments if s.get("language")})

                    audio_base64 = modal_res.get("audio_base64")
                    audio_bytes = base64.b64decode(audio_base64) if audio_base64 else None
                    audio_mime = modal_res.get("audio_mime", "audio/wav")

                # ============================================
                # DISPLAY SECTIONS IN EXACT ORDER
                # ============================================

                # 1. 🔍 Detected Languages
                st.subheader("🔍 Detected Languages")
                st.write(format_detected_languages(det_langs))

                # 2. 📝 Normalized Text
                st.subheader("📝 Normalized Text")
                st.write(norm_display_text)

                # 3. 🧠 Semantic Interpretation
                st.subheader("🧠 Semantic Interpretation")
                st.info(sem_display_text if sem_display_text else "No semantic interpretation available.")

                # 4. 🤖 MultiMix AI Response
                st.subheader("🤖 MultiMix AI Response")
                if response_text:
                    st.success(response_text)
                else:
                    st.warning("The AI model returned an empty response.")

                # 5. 🔊 AI Voice Response
                st.subheader("🔊 AI Voice Response")
                if audio_bytes:
                    st.audio(audio_bytes, format=audio_mime)
                    st.caption("Generated using IndicF5")
                else:
                    st.info("Voice audio output is not available for this response.")

                # ============================================
                # DEVELOPER DETAILS (COLLAPSED BY DEFAULT)
                # ============================================
                with st.expander("🛠️ Developer Details", expanded=False):
                    st.write("**Backend:**", "Modal GPU Backend (T4)" if deployment_mode != "FULL LOCAL MODEL" else "Local GPU/CPU")
                    st.write("**Language Segments:**")
                    for seg in segments:
                        st.write(
                            f"- `{seg.get('token', '')}` → **{seg.get('language', 'Unknown')}** "
                            f"(method: {seg.get('method', 'pattern')}, conf: {seg.get('confidence', 1.0):.2f})"
                        )

            except Exception as err:
                st.error(f"An error occurred while processing text: {err}")


# ============================================================
# TAB 2: VOICE INTERACTION
# ============================================================

with tab_voice:
    st.subheader("🎤 Voice Input")
    st.caption("Upload a voice recording to transcribe, analyze and receive an AI voice response.")

    audio_file = st.file_uploader(
        "Upload a voice recording",
        type=["wav", "mp3", "mp4", "m4a", "ogg", "flac"],
        key="uploader_voice",
    )

    if audio_file is not None:
        st.audio(audio_file, format=audio_file.type)

        if st.button("Process Voice", type="primary", use_container_width=True, key="btn_voice_submit"):
            # ----------------------------------------------------
            # LOCAL MODEL VOICE PROCESSING
            # ----------------------------------------------------
            if deployment_mode == "FULL LOCAL MODEL":
                if not model_status.get("whisper"):
                    st.info("ℹ️ Local Faster-Whisper model is not provisioned.")
                    st.stop()

                upload_dir = UPLOADS_AUDIO_DIR
                safe_filename = Path(audio_file.name).name
                save_audio_path = upload_dir / safe_filename
                save_audio_path.write_bytes(audio_file.getbuffer())

                try:
                    with st.spinner("Transcribing and analyzing voice locally..."):
                        result = process_voice(str(save_audio_path))

                    asr_quality = result.get("asr_quality", "good")
                    status = result.get("status", "")

                    if asr_quality == "failed" or status == "No speech detected" or "couldn't reliably understand" in str(status):
                        st.warning("Sorry, I couldn't reliably understand the audio. Please try recording again with clearer speech.")
                        with st.expander("🛠️ Developer Details", expanded=False):
                            st.write("**ASR Quality:**", asr_quality)
                            st.write("**Pipeline Status:**", status)
                            st.write("**Quality Reasons:**", result.get("asr_quality_reasons", []))
                            st.write("**Whisper Language:**", result.get("whisper_language", "unknown"))
                            st.write("**Raw Transcription:**", result.get("transcription", ""))
                        st.stop()

                    transcription = result.get("transcription", "")
                    whisper_lang = result.get("whisper_language", "unknown")
                    segments = result.get("segments", [])
                    response_text = result.get("response", "")

                    norm_raw = result.get("normalized", {})
                    norm_display_text = norm_raw.get("normalized_text", transcription) if isinstance(norm_raw, dict) else transcription

                    sem_raw = result.get("semantic", {})
                    sem_display_text = sem_raw.get("semantic_text", "") if isinstance(sem_raw, dict) else ""

                    audio_path = None
                    if response_text and model_status.get("indicf5"):
                        with st.spinner("Generating voice response..."):
                            try:
                                output_dir = GENERATED_AUDIO_DIR
                                voice_output = output_dir / f"response_{uuid.uuid4().hex}.wav"
                                audio_path = synthesize_speech(
                                    text=response_text,
                                    output_path=str(voice_output),
                                    speed=1.0,
                                )
                            except Exception as tts_err:
                                st.warning(f"Voice generation unavailable: {tts_err}")

                    audio_bytes = Path(audio_path).read_bytes() if (audio_path and Path(audio_path).exists()) else None
                    audio_mime = "audio/wav"

                finally:
                    save_audio_path.unlink(missing_ok=True)

            # ----------------------------------------------------
            # MODAL GPU BACKEND VOICE PROCESSING (CLOUD / DEMO MODE)
            # ----------------------------------------------------
            else:
                try:
                    with st.spinner("Transcribing and processing voice on Modal GPU (Whisper + Qwen + IndicF5)..."):
                        raw_audio_bytes = audio_file.getbuffer().tobytes()
                        audio_b64 = base64.b64encode(raw_audio_bytes).decode("utf-8")

                        payload = {
                            "mode": "voice",
                            "audio_base64": audio_b64,
                            "filename": audio_file.name,
                            "target_language": "Auto",
                        }
                        modal_res = call_modal_backend(payload)

                    if modal_res.get("status") == "error":
                        st.error(f"Backend Error: {modal_res.get('error', 'Unknown backend error')}")
                        st.stop()

                    asr_quality = modal_res.get("asr_quality", "good")
                    status = modal_res.get("status", "")

                    if asr_quality == "failed" or status == "No speech detected" or "couldn't reliably understand" in str(status):
                        st.warning("Sorry, I couldn't reliably understand the audio. Please try recording again with clearer speech.")
                        with st.expander("🛠️ Developer Details", expanded=False):
                            st.write("**ASR Quality:**", asr_quality)
                            st.write("**Pipeline Status:**", status)
                            st.write("**Quality Reasons:**", modal_res.get("asr_quality_reasons", []))
                            st.write("**Whisper Language:**", modal_res.get("whisper_language", "unknown"))
                            st.write("**Raw Transcription:**", modal_res.get("transcription", ""))
                        st.stop()

                    transcription = modal_res.get("transcription", "")
                    whisper_lang = modal_res.get("whisper_language", "unknown")
                    segments = modal_res.get("segments", [])
                    response_text = modal_res.get("response", "")

                    norm_raw = modal_res.get("normalized", {})
                    if isinstance(norm_raw, dict):
                        norm_display_text = norm_raw.get("normalized_text", transcription)
                    else:
                        norm_display_text = str(norm_raw) if norm_raw else transcription

                    sem_raw = modal_res.get("semantic", {})
                    if isinstance(sem_raw, dict):
                        sem_display_text = sem_raw.get("semantic_text", "")
                    else:
                        sem_display_text = str(sem_raw) if sem_raw else ""

                    audio_base64 = modal_res.get("audio_base64")
                    audio_bytes = base64.b64decode(audio_base64) if audio_base64 else None
                    audio_mime = modal_res.get("audio_mime", "audio/wav")

                except Exception as err:
                    st.error(f"An error occurred while processing voice: {err}")
                    st.stop()

            # ============================================
            # DISPLAY SECTIONS IN EXACT ORDER
            # ============================================

            # ASR Quality Banner
            if asr_quality == "good":
                st.success("🟢 Speech recognized successfully")
            elif asr_quality == "low":
                st.warning("🟡 Speech recognized with low confidence")

            # 1. 🎙️ Transcription
            st.subheader("🎙️ Transcription")
            st.write(transcription)

            # 2. 🌐 Whisper Language
            st.subheader("🌐 Whisper Language")
            st.write(LANG_NAME_MAP.get(str(whisper_lang).lower(), str(whisper_lang).title()))

            # 3. 🔍 Detected Languages
            st.subheader("🔍 Detected Languages")
            seg_langs = list({s["language"] for s in segments if s.get("language")})
            st.write(format_detected_languages(seg_langs))

            # 4. 📝 Normalized Text
            st.subheader("📝 Normalized Text")
            st.write(norm_display_text)

            # 5. 🧠 Semantic Interpretation
            st.subheader("🧠 Semantic Interpretation")
            st.info(sem_display_text if sem_display_text else "No semantic interpretation available.")

            # 6. 🤖 MultiMix AI Response
            st.subheader("🤖 MultiMix AI Response")
            if response_text:
                st.success(response_text)
            else:
                st.warning("No AI response was generated.")

            # 7. 🔊 AI Voice Response
            st.subheader("🔊 AI Voice Response")
            if audio_bytes:
                st.audio(audio_bytes, format=audio_mime)
                st.caption("Generated using IndicF5")
            else:
                st.info("Voice audio output is not available for this response.")

            # DEVELOPER DETAILS (COLLAPSED BY DEFAULT)
            with st.expander("🛠️ Developer Details", expanded=False):
                st.write("**Backend:**", "Modal GPU Backend (T4)" if deployment_mode != "FULL LOCAL MODEL" else "Local GPU/CPU")
                st.write("**ASR Quality Classification:**", asr_quality)
                st.write("**ASR Quality Reasons:**", modal_res.get("asr_quality_reasons", []) if deployment_mode != "FULL LOCAL MODEL" else result.get("asr_quality_reasons", []))
                st.write("**Pipeline Status:**", status)
                st.write("**Language Segments:**")
                for seg in segments:
                    st.write(
                        f"- `{seg.get('token', '')}` → **{seg.get('language', 'Unknown')}** "
                        f"(method: {seg.get('method', 'pattern')}, conf: {seg.get('confidence', 1.0):.2f})"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption("MultiMix AI — Multilingual Code-Mixed Language Understanding System")
