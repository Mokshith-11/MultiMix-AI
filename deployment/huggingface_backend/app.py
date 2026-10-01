import json

import gradio as gr
import spaces

from src.voice.pipeline import process_text, process_voice


LANGUAGES = [
    "Auto",
    "Telugu",
    "Tamil",
    "Hindi",
    "Bengali",
    "English",
]


def run_text(text, target_language):
    target = "" if target_language == "Auto" else target_language
    result = process_text(text, target)
    return (
        json.dumps(result, ensure_ascii=False, indent=2),
        result.get("response_audio_path"),
    )


def run_voice(audio, target_language):
    target = "" if target_language == "Auto" else target_language
    result = process_voice(audio, target)
    return (
        json.dumps(result, ensure_ascii=False, indent=2),
        result.get("response_audio_path"),
    )


@spaces.GPU(duration=180)
def text_inference(text, target_language):
    return run_text(text, target_language)


@spaces.GPU(duration=180)
def voice_inference(audio, target_language):
    return run_voice(audio, target_language)


with gr.Blocks(
    title="MultiMix AI",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
# MULTIMIX AI
### Multilingual & Code-Mixed Conversational AI

Supports:
**Telugu · Tamil · Hindi · Bengali · English**

Process:
**Input → Language Detection → Semantic Understanding → Qwen → Translation → IndicF5 Voice**
"""
    )

    with gr.Tab("Text Interaction"):
        text_input = gr.Textbox(
            label="Code-Mixed Text",
            placeholder=(
                "Nenu today college ki vellanu but my friend "
                "Tamil-la pesitu irindhan."
            ),
            lines=4,
        )

        text_language = gr.Dropdown(
            choices=LANGUAGES,
            value="Auto",
            label="Response Language",
        )

        text_button = gr.Button(
            "Process Text",
            variant="primary",
        )

        text_result = gr.Code(
            label="MultiMix AI Result",
            language="json",
        )

        text_audio = gr.Audio(
            label="AI Voice Response",
            type="filepath",
        )

        text_button.click(
            fn=text_inference,
            inputs=[text_input, text_language],
            outputs=[text_result, text_audio],
        )

    with gr.Tab("Voice Interaction"):
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Voice Input",
        )

        voice_language = gr.Dropdown(
            choices=LANGUAGES,
            value="Auto",
            label="Response Language",
        )

        voice_button = gr.Button(
            "Process Voice",
            variant="primary",
        )

        voice_result = gr.Code(
            label="MultiMix AI Result",
            language="json",
        )

        voice_audio = gr.Audio(
            label="AI Voice Response",
            type="filepath",
        )

        voice_button.click(
            fn=voice_inference,
            inputs=[audio_input, voice_language],
            outputs=[voice_result, voice_audio],
        )


demo.launch()
