import re
from typing import Dict, List, Set

from src.language.detector import (
    AMBIGUOUS_WORDS,
    CONTEXT_BIGRAMS,
    ENGLISH_WORDS,
    LANGUAGE_MARKERS,
    MORPHOLOGY_PATTERNS,
    ROMANIZED_LEXICON,
    SCRIPT_RANGES,
)


SEGMENT_LEXICON = ROMANIZED_LEXICON


# ============================================================
# NATIVE SCRIPT DETECTION
# ============================================================

def detect_native_script(token: str) -> str | None:
    """
    Detect language from native Unicode script.

    Telugu  : U+0C00-U+0C7F
    Tamil   : U+0B80-U+0BFF
    Bengali : U+0980-U+09FF
    Hindi   : U+0900-U+097F
    """
    for language, pattern in SCRIPT_RANGES.items():
        if pattern.search(token):
            return language
    return None


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_with_punctuation(text: str) -> List[str]:
    """
    Tokenize Romanized and native-script text, preserving
    explicit language markers such as Tamil-la, Telugu-lo.
    """
    return re.findall(
        r"[A-Za-z\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F]+(?:-[A-Za-z]+)?",
        text,
    )


# ============================================================
# SINGLE TOKEN DETECTION
# ============================================================

def detect_token_language(token: str) -> Dict:
    """
    Detect the language of an isolated single token.

    Detection priority:
    1. Native Unicode script
    2. Explicit language marker (e.g. Tamil-la)
    3. Romanized Indic lexicon
    4. English lexicon
    5. Morphological suffix patterns
    6. Ambiguous-word handling
    7. Unknown
    """
    original = token
    lowered = token.lower().strip()

    # 1. Native script
    native_lang = detect_native_script(original)
    if native_lang:
        return {
            "token": original,
            "language": native_lang,
            "method": "native_script",
            "confidence": 1.0,
        }

    # 2. Explicit language marker
    parts = lowered.split("-")
    for part in parts:
        if part in LANGUAGE_MARKERS:
            return {
                "token": original,
                "language": LANGUAGE_MARKERS[part],
                "method": "language_marker",
                "confidence": 1.0,
            }

    # 3. Romanized lexicon
    candidates = set(parts)
    candidates.add(lowered)
    matches = {}
    for language, words in SEGMENT_LEXICON.items():
        matched = candidates.intersection(words)
        if matched:
            score = 0
            for word in matched:
                score += 2 if len(word) >= 5 else 1
            if score > 0:
                matches[language] = score

    if matches:
        best_language = max(matches, key=matches.get)
        best_score = matches[best_language]
        return {
            "token": original,
            "language": best_language,
            "method": "romanized_lexicon",
            "confidence": min(1.0, best_score / 2.0),
        }

    # 4. English lexicon
    if lowered in ENGLISH_WORDS:
        return {
            "token": original,
            "language": "English",
            "method": "english_lexicon",
            "confidence": 1.0,
        }

    # 5. Morphology pattern
    for language, patterns in MORPHOLOGY_PATTERNS.items():
        for pat in patterns:
            if pat.match(lowered):
                return {
                    "token": original,
                    "language": language,
                    "method": "morphology",
                    "confidence": 0.85,
                }

    # 6. Ambiguous token
    if lowered in AMBIGUOUS_WORDS:
        return {
            "token": original,
            "language": "Unknown",
            "method": "ambiguous",
            "confidence": 0.0,
        }

    # 7. Unknown
    return {
        "token": original,
        "language": "Unknown",
        "method": "unknown",
        "confidence": 0.0,
    }


# ============================================================
# CONTEXT RESOLUTION FOR AMBIGUOUS TOKENS
# ============================================================

def resolve_ambiguous_tokens(segments: List[Dict]) -> List[Dict]:
    """
    Resolve ambiguous tokens using nearby confident language predictions.
    """
    resolved = []
    indian_languages = {"Telugu", "Tamil", "Hindi", "Bengali"}

    for index, segment in enumerate(segments):
        if segment["language"] != "Unknown":
            resolved.append(segment)
            continue

        token = segment["token"].lower()
        if token not in AMBIGUOUS_WORDS:
            resolved.append(segment)
            continue

        previous_language = None
        next_language = None

        # Look backward for nearest known language
        for i in range(index - 1, -1, -1):
            lang = segments[i]["language"]
            if lang != "Unknown":
                previous_language = lang
                break

        # Look forward for nearest known language
        for i in range(index + 1, len(segments)):
            lang = segments[i]["language"]
            if lang != "Unknown":
                next_language = lang
                break

        # Both sides agree on an Indian language
        if (
            previous_language
            and next_language
            and previous_language == next_language
            and previous_language != "English"
        ):
            seg_copy = segment.copy()
            seg_copy["language"] = previous_language
            seg_copy["method"] = "context_resolution"
            seg_copy["confidence"] = 0.85
            resolved.append(seg_copy)

        # Previous token is an Indian language
        elif previous_language in indian_languages:
            seg_copy = segment.copy()
            seg_copy["language"] = previous_language
            seg_copy["method"] = "context_resolution"
            seg_copy["confidence"] = 0.70
            resolved.append(seg_copy)

        # Next token is an Indian language
        elif next_language in indian_languages:
            seg_copy = segment.copy()
            seg_copy["language"] = next_language
            seg_copy["method"] = "context_resolution"
            seg_copy["confidence"] = 0.70
            resolved.append(seg_copy)

        else:
            resolved.append(segment)

    return resolved


# ============================================================
# SEQUENCE-AWARE CODE-MIXED SEGMENTATION
# ============================================================

def segment_code_mix(text: str) -> List[Dict]:
    """
    Detect language for every token in a multilingual code-mixed sentence.
    Integrates sequence-level signals:
    - Collocation / Bigram context
    - Name-marker neutralization
    - Lexical, morphological, and script detection
    - Ambiguous-word context resolution
    """
    tokens = tokenize_with_punctuation(text)
    tokens_lower = [t.lower() for t in tokens]
    segments = []

    # Identify name candidates (e.g. word directly following peru, peyar, naam, name)
    name_indices: Set[int] = set()
    for idx, tok in enumerate(tokens_lower):
        if tok in {"peru", "peyar", "naam", "name"} and idx + 1 < len(tokens_lower):
            name_indices.add(idx + 1)

    # Context bigrams map: assigns high confidence to recognized collocations
    bigram_lang: Dict[int, str] = {}
    for i in range(len(tokens_lower) - 1):
        bg = (tokens_lower[i], tokens_lower[i + 1])
        for lang, bigrams in CONTEXT_BIGRAMS.items():
            if bg in bigrams:
                bigram_lang[i] = lang
                bigram_lang[i + 1] = lang

    for idx, (orig, tok) in enumerate(zip(tokens, tokens_lower)):
        # 1. Native Unicode script
        native_lang = detect_native_script(orig)
        if native_lang:
            segments.append({
                "token": orig,
                "language": native_lang,
                "method": "native_script",
                "confidence": 1.0,
            })
            continue

        # 2. Proper name candidate following name marker
        if idx in name_indices and tok not in ENGLISH_WORDS and tok not in AMBIGUOUS_WORDS:
            is_in_lexicon = any(tok in words for words in SEGMENT_LEXICON.values())
            if not is_in_lexicon:
                segments.append({
                    "token": orig,
                    "language": "Unknown",
                    "method": "proper_name",
                    "confidence": 0.0,
                })
                continue

        # 3. Explicit language marker (e.g. Tamil-la)
        parts = tok.split("-")
        marker_lang = None
        for part in parts:
            if part in LANGUAGE_MARKERS:
                marker_lang = LANGUAGE_MARKERS[part]
                break
        if marker_lang:
            segments.append({
                "token": orig,
                "language": marker_lang,
                "method": "language_marker",
                "confidence": 1.0,
            })
            continue

        # 4. Bigram collocation match
        if idx in bigram_lang:
            segments.append({
                "token": orig,
                "language": bigram_lang[idx],
                "method": "context_collocation",
                "confidence": 0.90,
            })
            continue

        # 5. Romanized lexicon
        matches = {}
        for language, words in SEGMENT_LEXICON.items():
            if tok in words:
                matches[language] = 2.0 if len(tok) >= 5 else 1.5
        if matches:
            best_lang = max(matches, key=matches.get)
            segments.append({
                "token": orig,
                "language": best_lang,
                "method": "romanized_lexicon",
                "confidence": 0.95,
            })
            continue

        # 6. English lexicon
        if tok in ENGLISH_WORDS:
            segments.append({
                "token": orig,
                "language": "English",
                "method": "english_lexicon",
                "confidence": 1.0,
            })
            continue

        # 7. Morphology pattern
        morph_lang = None
        for language, patterns in MORPHOLOGY_PATTERNS.items():
            for pat in patterns:
                if pat.match(tok):
                    morph_lang = language
                    break
            if morph_lang:
                break
        if morph_lang:
            segments.append({
                "token": orig,
                "language": morph_lang,
                "method": "morphology",
                "confidence": 0.85,
            })
            continue

        # 8. Ambiguous word
        if tok in AMBIGUOUS_WORDS:
            segments.append({
                "token": orig,
                "language": "Unknown",
                "method": "ambiguous",
                "confidence": 0.0,
            })
            continue

        # 9. Unknown
        segments.append({
            "token": orig,
            "language": "Unknown",
            "method": "unknown",
            "confidence": 0.0,
        })

    # Resolve ambiguous tokens using context
    segments = resolve_ambiguous_tokens(segments)

    return segments


# ============================================================
# PUBLIC LANGUAGE SEGMENT API
# ============================================================

def get_language_segments(text: str) -> List[Dict]:
    """
    Return token-level language segments.
    """
    return segment_code_mix(text)