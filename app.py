import tempfile
import uuid
from pathlib import Path

import streamlit as st

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
        st.info("☁️ **Free Cloud Demo**")

st.caption("Understand and respond to multilingual Indian code-mixed text and voice.")

if deployment_mode != "FULL LOCAL MODEL":
    st.info(
        "ℹ️ **Streamlit Cloud Demo Mode Active**: Linguistic analysis, token segmentation, normalization, "
        "and semantic interpretation run in pure software mode. Heavy model weights (Qwen, Faster-Whisper, IndicF5) "
        "are not provisioned in this free cloud demo."
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
                # Progress Step 1: Text Analysis
                with st.spinner("Analyzing multilingual input..."):
                    detection = analyze_code_mix(input_text)
                    segments = get_language_segments(input_text)
                    normalized = normalize_text(text=input_text, segments=segments)
                    semantic = interpret_code_mix(text=input_text, segments=segments)

                # Collect detected languages
                det_langs = detection.get("languages", [])
                if not det_langs and segments:
                    det_langs = list({s["language"] for s in segments if s.get("language")})

                # Progress Step 2: AI Response
                response_text = ""
                if model_status.get("qwen"):
                    with st.spinner("Generating AI response..."):
                        response_text = generate_response(
                            text=input_text,
                            segments=segments,
                            semantic_input=semantic.get("semantic_text", input_text),
                            max_new_tokens=100,
                        )

                # Progress Step 3: Voice Response
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

                # ============================================
                # DISPLAY SECTIONS IN EXACT ORDER
                # ============================================

                # 1. 🔍 Detected Languages
                st.subheader("🔍 Detected Languages")
                st.write(format_detected_languages(det_langs))

                # 2. 📝 Normalized Text
                st.subheader("📝 Normalized Text")
                st.write(normalized.get("normalized_text", input_text))

                # 3. 🧠 Semantic Interpretation
                st.subheader("🧠 Semantic Interpretation")
                st.info(semantic.get("semantic_text", ""))

                # 4. 🤖 MultiMix AI Response
                st.subheader("🤖 MultiMix AI Response")
                if response_text:
                    st.success(response_text)
                elif not model_status.get("qwen"):
                    st.info("ℹ️ Local Qwen model is not provisioned in this free cloud demo.")
                else:
                    st.warning("The AI model returned an empty response.")

                # 5. 🔊 AI Voice Response
                st.subheader("🔊 AI Voice Response")
                if audio_path and Path(audio_path).exists():
                    st.audio(audio_path, format="audio/wav")
                    st.caption("Generated using IndicF5")
                elif not model_status.get("indicf5"):
                    st.info("ℹ️ AI voice response is unavailable in the free cloud demo because IndicF5 is not provisioned.")
                else:
                    st.info("Voice audio output is not available for this response.")

                # ============================================
                # DEVELOPER DETAILS (COLLAPSED BY DEFAULT)
                # ============================================
                with st.expander("🛠️ Developer Details", expanded=False):
                    st.write("**IndicLID Model Active:**", "Yes" if detection.get("indiclid_available") else "No (Fallback Mode)")
                    st.write("**Language Count:**", detection.get("language_count", len(det_langs)))
                    st.write("**Code-Mixed:**", "Yes" if detection.get("is_code_mixed") else "No")
                    st.write("**Language Segments:**")
                    for seg in segments:
                        st.write(
                            f"- `{seg['token']}` → **{seg['language']}** "
                            f"(method: {seg['method']}, conf: {seg['confidence']:.2f})"
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
            if not model_status.get("whisper"):
                st.info(
                    "ℹ️ Voice transcription and processing requires the local Faster-Whisper model, "
                    "which is not provisioned in this free cloud demo."
                )
                st.stop()

            upload_dir = UPLOADS_AUDIO_DIR
            safe_filename = Path(audio_file.name).name
            save_audio_path = upload_dir / safe_filename
            save_audio_path.write_bytes(audio_file.getbuffer())

            try:
                # Progress Step 1: Transcribe and analyze voice
                with st.spinner("Transcribing and analyzing voice..."):
                    result = process_voice(str(save_audio_path))

                asr_quality = result.get("asr_quality", "good")
                status = result.get("status", "")

                # ============================================
                # FAILED ASR HANDLING
                # ============================================
                if (
                    asr_quality == "failed"
                    or status == "No speech detected"
                    or "couldn't reliably understand" in str(status)
                ):
                    st.warning("Sorry, I couldn't reliably understand the audio. Please try recording again with clearer speech.")

                    with st.expander("🛠️ Developer Details", expanded=False):
                        st.write("**ASR Quality:**", asr_quality)
                        st.write("**Pipeline Status:**", status)
                        st.write("**Quality Reasons:**", result.get("asr_quality_reasons", []))
                        st.write("**Whisper Language:**", result.get("whisper_language", "unknown"))
                        st.write("**Raw Transcription:**", result.get("transcription", ""))

                else:
                    # ASR Quality Status Banner
                    if asr_quality == "good":
                        st.success("🟢 Speech recognized successfully")
                    elif asr_quality == "low":
                        st.warning("🟡 Speech recognized with low confidence")

                    response_text = result.get("response", "")
                    audio_path = None

                    if response_text and model_status.get("indicf5"):
                        # Progress Step 2: Voice response generation
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

                    # ============================================
                    # DISPLAY SECTIONS IN EXACT ORDER
                    # ============================================

                    # 1. 🎙️ Transcription
                    st.subheader("🎙️ Transcription")
                    st.write(result.get("transcription", ""))

                    # 2. 🌐 Whisper Language
                    st.subheader("🌐 Whisper Language")
                    whisper_lang = result.get("whisper_language", "unknown")
                    st.write(LANG_NAME_MAP.get(str(whisper_lang).lower(), str(whisper_lang).title()))

                    # 3. 🔍 Detected Languages
                    st.subheader("🔍 Detected Languages")
                    seg_langs = list({s["language"] for s in result.get("segments", []) if s.get("language")})
                    st.write(format_detected_languages(seg_langs))

                    # 4. 📝 Normalized Text
                    st.subheader("📝 Normalized Text")
                    norm_text = result.get("normalized", {}).get("normalized_text", "")
                    st.write(norm_text if norm_text else result.get("transcription", ""))

                    # 5. 🧠 Semantic Interpretation
                    st.subheader("🧠 Semantic Interpretation")
                    sem_text = result.get("semantic", {}).get("semantic_text", "")
                    st.info(sem_text if sem_text else "No semantic interpretation available.")

                    # 6. 🤖 MultiMix AI Response
                    st.subheader("🤖 MultiMix AI Response")
                    if response_text:
                        st.success(response_text)
                    elif not model_status.get("qwen"):
                        st.info("ℹ️ Local Qwen model is not provisioned in this free cloud demo.")
                    elif result.get("response_error"):
                        st.warning("AI response generation encountered an issue.")
                    else:
                        st.warning("No AI response was generated.")

                    # 7. 🔊 AI Voice Response
                    st.subheader("🔊 AI Voice Response")
                    if audio_path and Path(audio_path).exists():
                        st.audio(audio_path, format="audio/wav")
                        st.caption("Generated using IndicF5")
                    elif not model_status.get("indicf5"):
                        st.info("ℹ️ AI voice response is unavailable in the free cloud demo because IndicF5 is not provisioned.")
                    else:
                        st.info("Voice audio output is not available for this response.")

                    # DEVELOPER DETAILS (COLLAPSED BY DEFAULT)
                    with st.expander("🛠️ Developer Details", expanded=False):
                        st.write("**IndicLID Model Active:**", "Yes" if result.get("indiclid_available", True) else "No (Fallback Mode)")
                        st.write("**ASR Quality Classification:**", asr_quality)
                        st.write("**ASR Quality Reasons:**", result.get("asr_quality_reasons", []))
                        st.write("**Pipeline Status:**", status)
                        st.write("**Language Segments:**")
                        for seg in result.get("segments", []):
                            st.write(
                                f"- `{seg['token']}` → **{seg['language']}** "
                                f"(method: {seg['method']}, conf: {seg['confidence']:.2f})"
                            )

            except Exception as err:
                st.error(f"An error occurred while processing voice: {err}")

            finally:
                save_audio_path.unlink(missing_ok=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption("MultiMix AI — Multilingual Code-Mixed Language Understanding System")
