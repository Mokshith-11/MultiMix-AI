from __future__ import annotations

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

from src.config import (
    M2M100_MODEL_ID,
    TELUGU_MODEL_ID,
)

_M2M_TOKENIZER = None
_M2M_MODEL = None

_TELUGU_TOKENIZER = None
_TELUGU_MODEL = None


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _load_m2m100():
    global _M2M_TOKENIZER, _M2M_MODEL

    if _M2M_TOKENIZER is not None and _M2M_MODEL is not None:
        return _M2M_TOKENIZER, _M2M_MODEL

    print("Loading M2M100 translation model...")

    _M2M_TOKENIZER = M2M100Tokenizer.from_pretrained(
        M2M100_MODEL_ID
    )

    _M2M_MODEL = M2M100ForConditionalGeneration.from_pretrained(
        M2M100_MODEL_ID,
        torch_dtype=_dtype(),
    )

    _M2M_MODEL = _M2M_MODEL.to(_device())
    _M2M_MODEL.eval()

    print("M2M100 loaded.")

    return _M2M_TOKENIZER, _M2M_MODEL


def _load_telugu():
    global _TELUGU_TOKENIZER, _TELUGU_MODEL

    if _TELUGU_TOKENIZER is not None and _TELUGU_MODEL is not None:
        return _TELUGU_TOKENIZER, _TELUGU_MODEL

    print("Loading English-to-Telugu model...")

    _TELUGU_TOKENIZER = AutoTokenizer.from_pretrained(
        TELUGU_MODEL_ID
    )

    _TELUGU_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
        TELUGU_MODEL_ID,
        torch_dtype=_dtype(),
    )

    _TELUGU_MODEL = _TELUGU_MODEL.to(_device())
    _TELUGU_MODEL.eval()

    print("English-to-Telugu model loaded.")

    return _TELUGU_TOKENIZER, _TELUGU_MODEL


def translate_to_indic(
    text: str,
    target_language: str,
) -> str:
    """
    Translate an English Qwen response into the requested
    Indian language.

    Supported:
        English
        Telugu
        Tamil
        Hindi
        Bengali
    """

    text = (text or "").strip()

    if not text:
        raise ValueError("Cannot translate empty text.")

    target = target_language.strip().lower()

    if target in {"english", "en"}:
        return text

    # ---------------------------------------------------------
    # Telugu
    # ---------------------------------------------------------

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
            forced_bos_id = tokenizer.lang_code_to_id.get(
                "tel_Telu"
            )

        if forced_bos_id is None and hasattr(
            tokenizer,
            "convert_tokens_to_ids",
        ):
            forced_bos_id = tokenizer.convert_tokens_to_ids(
                "tel_Telu"
            )

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

    # ---------------------------------------------------------
    # Hindi / Tamil / Bengali
    # ---------------------------------------------------------

    language_map = {
        "hindi": "hi",
        "hi": "hi",
        "tamil": "ta",
        "ta": "ta",
        "bengali": "bn",
        "bn": "bn",
    }

    if target not in language_map:
        raise ValueError(
            f"Unsupported target language: {target_language}"
        )

    tokenizer, model = _load_m2m100()

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

    forced_bos_id = tokenizer.get_lang_id(
        language_map[target]
    )

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
