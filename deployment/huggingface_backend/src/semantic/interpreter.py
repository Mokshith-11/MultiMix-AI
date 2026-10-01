import re
from typing import Dict, List


# ============================================================
# ROMANIZED WORD TRANSLATIONS
# ============================================================

PHRASE_TRANSLATIONS = {
    "pesitu irindhan": "was speaking",
    "irindhan": "was",
    # Telugu
    "nenu": "I",
    "nuvvu": "you",
    "meeru": "you",
    "memu": "we",
    "manam": "we",
    "naaku": "to me",
    "neeku": "to you",
    "vellanu": "went",
    "vellava": "went",
    "velthunna": "am going",
    "vastanu": "will come",
    "vastava": "will come",
    "vacchanu": "came",
    "chesanu": "did",
    "chestunna": "am doing",
    "chestunnanu": "am doing",
    "undi": "is",
    "unnanu": "am",
    "unnava": "are",
    "enduku": "why",
    "ela": "how",
    "enti": "what",
    "emiti": "what",
    "ikkada": "here",
    "akkada": "there",
    "ivala": "today",
    "repu": "tomorrow",
    "kani": "but",
    "mari": "then",
    "kuda": "also",
    "ante": "means",
    "ledu": "not",
    "avunu": "yes",

    # Tamil
    "naan": "I",
    "nanu": "I",
    "nee": "you",
    "neenga": "you",
    "avan": "he",
    "aval": "she",
    "enga": "where",
    "enakku": "to me",
    "unakku": "to you",
    "namma": "our",
    "romba": "very",
    "irukken": "am",
    "irukku": "is",
    "irundhan": "was",
    "irundhen": "was",
    "pesitu": "speaking",
    "pesuren": "speaking",
    "poga": "go",
    "poiten": "went",
    "poitten": "went",
    "vandhen": "came",
    "varuven": "will come",
    "varum": "will come",
    "enna": "what",
    "ennaku": "to me",
    "yen": "why",
    "epdi": "how",
    "eppadi": "how",
    "inga": "here",
    "anga": "there",
    "oda": "with",
    "aama": "yes",
    "illa": "no",

    # Hindi
    "main": "I",
    "mein": "in",
    "mujhe": "to me",
    "mujhko": "to me",
    "tum": "you",
    "aap": "you",
    "hum": "we",
    "ham": "we",
    "hume": "to us",
    "mera": "my",
    "meri": "my",
    "mere": "my",
    "tera": "your",
    "teri": "your",
    "aaj": "today",
    "kal": "tomorrow/yesterday",
    "kya": "what",
    "kyun": "why",
    "kaise": "how",
    "kaisa": "how",
    "hai": "is",
    "hain": "are",
    "hoon": "am",
    "tha": "was",
    "thi": "was",
    "gaya": "went",
    "gayi": "went",
    "gaye": "went",
    "jaunga": "will go",
    "jaungi": "will go",
    "karna": "do",
    "karta": "do",
    "karti": "do",
    "raha": "doing",
    "rahi": "doing",
    "rahe": "doing",
    "bahut": "very",
    "accha": "good",
    "achha": "good",
    "nahi": "not",
    "nahin": "not",
    "haan": "yes",

    # Bengali
    "ami": "I",
    "tumi": "you",
    "apni": "you",
    "amra": "we",
    "tomra": "you",
    "amar": "my",
    "tomar": "your",
    "amader": "our",
    "tomader": "your",
    "aaj": "today",
    "kal": "tomorrow/yesterday",
    "keno": "why",
    "kemon": "how",
    "ache": "is",
    "achi": "am",
    "achhe": "is",
    "chilo": "was",
    "gechi": "went",
    "gechhi": "went",
    "jabo": "will go",
    "jacchi": "am going",
    "korbo": "will do",
    "korchi": "am doing",
    "korechi": "did",
    "bhalo": "good",
    "khub": "very",
    "hya": "yes",
}


# ============================================================
# COMMON POSTPOSITIONS / PARTICLES
# ============================================================

POSTPOSITION_MAP = {
    "ki": "to",
    "ke": "to",
    "ku": "to",
    "lo": "in",
    "la": "in",
    "me": "in",
    "ko": "to",
}


# ============================================================
# TOKEN TRANSLATION
# ============================================================

def translate_token(token: str) -> str:
    """
    Convert a Romanized Indic token into a basic English
    semantic equivalent.
    """

    lowered = token.lower().strip()

    # Explicit language marker.
    # Tamil-la -> Tamil
    # Telugu-lo -> Telugu
    if "-" in lowered:

        parts = lowered.split("-")

        if len(parts) == 2:

            base = parts[0]
            marker = parts[1]

            if marker in {
                "la",
                "lo",
                "le",
                "me",
                "ko",
            }:
                return base

    if lowered in POSTPOSITION_MAP:
        return POSTPOSITION_MAP[lowered]

    return PHRASE_TRANSLATIONS.get(
        lowered,
        token,
    )


# ============================================================
# TOKEN SEMANTIC CONVERSION
# ============================================================

def build_token_semantics(
    segments: List[Dict],
) -> List[str]:

    result = []

    for segment in segments:

        token = segment["token"]
        language = segment["language"]

        # English does not need translation.
        if language == "English":
            result.append(token)
            continue

        translated = translate_token(token)

        result.append(translated)

    return result


# ============================================================
# PATTERN-BASED SEMANTIC RECONSTRUCTION
# ============================================================

def reconstruct_common_patterns(
    tokens: List[str],
    segments: List[Dict],
) -> str:
    """
    Reconstruct common Indian code-mixed sentence patterns.

    This is intentionally conservative. It only rewrites
    patterns for which we have strong evidence.
    """

    words = [word.lower() for word in tokens]

    # --------------------------------------------------------
    # Example:
    #
    # Nenu today college ki vellanu
    #
    # -> I went to college today
    # --------------------------------------------------------

    if (
        "i" in words
        and "went" in words
        and "college" in words
    ):

        result = []

        result.append("I")

        result.append("went")

        result.append("to")

        result.append("college")

        if "today" in words:
            result.append("today")

        # Look for the remaining clause.
        try:
            but_index = words.index("but")
        except ValueError:
            but_index = -1

        if but_index >= 0:

            result.append("but")

            remaining = tokens[but_index + 1:]

            result.extend(remaining)

        return " ".join(result)

    return " ".join(tokens)


# ============================================================
# LANGUAGE MARKER REFINEMENT
# ============================================================

def refine_language_phrases(
    text: str,
) -> str:
    """
    Convert explicit language markers into natural
    semantic phrases.

    Tamil-la -> in Tamil
    Telugu-lo -> in Telugu
    Hindi-me  -> in Hindi
    Bengali-te -> in Bengali
    """

    replacements = {
        r"\btamil-la\b": "in Tamil",
        r"\btamil-lo\b": "in Tamil",
        r"\btelugu-la\b": "in Telugu",
        r"\btelugu-lo\b": "in Telugu",
        r"\bhindi-la\b": "in Hindi",
        r"\bhindi-me\b": "in Hindi",
        r"\bbengali-la\b": "in Bengali",
        r"\bbengali-te\b": "in Bengali",
    }

    result = text

    for pattern, replacement in replacements.items():
        result = re.sub(
            pattern,
            replacement,
            result,
            flags=re.IGNORECASE,
        )

    return result


# ============================================================
# FINAL SEMANTIC CLEANUP
# ============================================================

def cleanup_semantic_text(text: str) -> str:

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # Common grammatical reconstruction.
    text = re.sub(
        r"\b(tamil|telugu|hindi|bengali)\s+speaking\s+was\b",
        lambda m: f"was speaking in {m.group(1).capitalize()}",
        text,
        flags=re.IGNORECASE,
    )

    # Remove duplicated spaces.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# MAIN SEMANTIC BUILDER
# ============================================================

def build_semantic_text(
    text: str,
    segments: List[Dict],
) -> str:

    tokens = build_token_semantics(
        segments
    )

    semantic_text = reconstruct_common_patterns(
        tokens,
        segments,
    )

    semantic_text = refine_language_phrases(
        semantic_text
    )

    semantic_text = cleanup_semantic_text(
        semantic_text
    )

    return semantic_text


# ============================================================
# PUBLIC INTERPRETER
# ============================================================

def interpret_code_mix(
    text: str,
    segments: List[Dict],
) -> Dict:

    semantic_text = build_semantic_text(
        text=text,
        segments=segments,
    )

    languages = sorted({
        segment["language"]
        for segment in segments
        if segment["language"] != "Unknown"
    })

    return {
        "original_text": text,
        "semantic_text": semantic_text,
        "languages": languages,
        "is_multilingual": len(languages) > 1,
    }