import re
from typing import Dict, List


# ============================================================
# BASIC NORMALIZATION RULES
# ============================================================

NORMALIZATION_RULES = {
    "tmrw": "tomorrow",
    "tomo": "tomorrow",
    "todayy": "today",
    "frnd": "friend",
    "clg": "college",
    "collg": "college",
    "pls": "please",
    "plz": "please",
    "bcoz": "because",
    "becoz": "because",
}


# ============================================================
# TOKEN NORMALIZATION
# ============================================================

def normalize_token(token: str) -> str:
    """
    Normalize a token while preserving its original
    multilingual identity.
    """

    original = token
    lowered = token.lower()

    # Preserve explicit language markers.
    # Example: Tamil-la, Telugu-lo
    if "-" in lowered:
        return original

    # Common abbreviations / spelling corrections.
    if lowered in NORMALIZATION_RULES:
        return NORMALIZATION_RULES[lowered]

    # Reduce excessive repeated characters.
    normalized = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        lowered,
    )

    return normalized


# ============================================================
# NORMALIZE SEGMENTS
# ============================================================

def normalize_segments(
    segments: List[Dict],
) -> List[Dict]:
    """
    Normalize detected language segments while preserving
    the original detection metadata.
    """

    normalized_segments = []

    for segment in segments:

        normalized_token = normalize_token(
            segment["token"]
        )

        normalized_segments.append(
            {
                "token": segment["token"],
                "normalized": normalized_token,
                "language": segment["language"],
                "method": segment["method"],
                "confidence": segment["confidence"],
            }
        )

    return normalized_segments


# ============================================================
# BUILD NORMALIZED TEXT
# ============================================================

def build_normalized_text(
    normalized_segments: List[Dict],
) -> str:
    """
    Build normalized text from normalized segments.
    """

    return " ".join(
        segment["normalized"]
        for segment in normalized_segments
    )


# ============================================================
# LANGUAGE-ANNOTATED TEXT
# ============================================================

def build_language_annotated_text(
    segments: List[Dict],
) -> str:
    """
    Build a readable language-aware representation.

    Example:
    Nenu [Telugu] today [English] college [English]
    ki [Telugu] vellanu [Telugu]
    """

    annotated = []

    for segment in segments:

        token = segment["token"]
        language = segment["language"]

        annotated.append(
            f"{token} [{language}]"
        )

    return " ".join(annotated)


# ============================================================
# SEMANTIC INPUT
# ============================================================

def build_semantic_input(
    normalized_segments: List[Dict],
) -> str:
    """
    Prepare a language-aware input representation for the
    response-generation layer.

    This does NOT claim to translate Romanized Indic text.
    It preserves the detected language information so that
    the response model receives useful context.
    """

    semantic_parts = []

    for segment in normalized_segments:

        token = segment["normalized"]
        language = segment["language"]

        if language == "Unknown":
            semantic_parts.append(token)
        else:
            semantic_parts.append(
                f"{token} [{language}]"
            )

    return " ".join(semantic_parts)


# ============================================================
# COMPLETE NORMALIZATION PIPELINE
# ============================================================

def normalize_text(
    text: str,
    segments: List[Dict],
) -> Dict:
    """
    Produce all normalized representations required by
    downstream MultiMix AI components.
    """

    normalized_segments = normalize_segments(
        segments
    )

    normalized_text = build_normalized_text(
        normalized_segments
    )

    annotated_text = build_language_annotated_text(
        segments
    )

    semantic_input = build_semantic_input(
        normalized_segments
    )

    return {
        "original_text": text,
        "normalized_text": normalized_text,
        "semantic_input": semantic_input,
        "language_annotated_text": annotated_text,
        "segments": normalized_segments,
    }