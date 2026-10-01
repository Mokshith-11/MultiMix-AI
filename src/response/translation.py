"""
MultiMix AI - Response Translation Layer

Qwen produces a grounded English response.
This module converts that response into an Indic target language
before sending it to IndicF5.

Supported:
    English -> Telugu
    English -> Hindi
    English -> Tamil
    English -> Bengali

Telugu:
    ai4bharat-specific English -> Telugu model

Hindi/Tamil/Bengali:
    facebook/m2m100_418M
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
)


M2M100_MODEL_ID = "facebook/m2m100_418M"
TELUGU_MODEL_ID = "anithasoma/nllb-finetuned-telugu"

M2M100_LOCAL_PATH = Path("models/m2m100-418m")
TELUGU_LOCAL_PATH = Path("models/english-to-telugu")

_M2M_TOKENIZER = None
_M2M_MODEL = None

_TELUGU_TOKENIZER = None
_TELUGU_MODEL = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_m2m100():
    global _M2M_TOKENIZER, _M2M_MODEL

    if _M2M_TOKENIZER is not None and _M2M_MODEL is not None:
        return _M2M_TOKENIZER, _M2M_MODEL

    local_path = Path(M2M100_LOCAL_PATH)

    model_source = (
        str(local_path)
        if (local_path / "config.json").exists()
        else M2M100_MODEL_ID
    )

    _M2M_TOKENIZER = M2M100Tokenizer.from_pretrained(
        model_source,
        local_files_only=local_path.exists(),
    )

    _M2M_MODEL = M2M100ForConditionalGeneration.from_pretrained(
        model_source,
        local_files_only=local_path.exists(),
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    ).to(_device())

    _M2M_MODEL.eval()

    return _M2M_TOKENIZER, _M2M_MODEL


def _load_telugu():
    global _TELUGU_TOKENIZER, _TELUGU_MODEL

    if _TELUGU_TOKENIZER is not None and _TELUGU_MODEL is not None:
        return _TELUGU_TOKENIZER, _TELUGU_MODEL

    local_path = Path(TELUGU_LOCAL_PATH)

    model_source = (
        str(local_path)
        if (local_path / "config.json").exists()
        else TELUGU_MODEL_ID
    )

    _TELUGU_TOKENIZER = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=local_path.exists(),
    )

    _TELUGU_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
        model_source,
        local_files_only=local_path.exists(),
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    ).to(_device())

    _TELUGU_MODEL.eval()

    return _TELUGU_TOKENIZER, _TELUGU_MODEL


def translate_to_indic(text: str, target_language: str) -> str:
    """
    Translate English response into an Indic target language.

    English is returned unchanged.
    """

    text = text.strip()

    if not text:
        raise ValueError("Cannot translate empty text.")

    target = target_language.strip().lower()

    if target in {"english", "en"}:
        return text

    # -----------------------------------------------------
    # Telugu
    # -----------------------------------------------------
    if target in {"telugu", "te"}:
        tokenizer, model = _load_telugu()

        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = "eng_Latn"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        device = next(model.parameters()).device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        forced_bos_id = None

        if hasattr(tokenizer, "lang_code_to_id"):
            forced_bos_id = tokenizer.lang_code_to_id.get("tel_Telu")

        if forced_bos_id is None:
            forced_bos_id = tokenizer.convert_tokens_to_ids("tel_Telu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=256,
                num_beams=4,
            )

        return tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0].strip()

    # -----------------------------------------------------
    # Bengali / Hindi / Tamil
    # -----------------------------------------------------
    lang_map = {
        "bengali": "bn",
        "bn": "bn",
        "hindi": "hi",
        "hi": "hi",
        "tamil": "ta",
        "ta": "ta",
    }

    if target not in lang_map:
        raise ValueError(
            f"Unsupported target language: {target_language}"
        )

    tokenizer, model = _load_m2m100()

    target_code = lang_map[target]

    tokenizer.src_lang = "en"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    device = next(model.parameters()).device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    forced_bos_id = tokenizer.get_lang_id(target_code)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            max_new_tokens=256,
            num_beams=4,
        )

    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )[0].strip()
