import re
from typing import Dict, List


# ============================================================
# ROMANIZED LANGUAGE LEXICON
# ============================================================

SEGMENT_LEXICON = {
    "Telugu": {
        "nenu", "nuvvu", "meeru", "memu", "manam",
        "naaku", "neeku", "vellanu", "vellava",
        "velthunna", "vastanu", "vastava", "vacchanu",
        "chesanu", "chestunna", "chestunnanu",
        "undi", "unnanu", "unnava", "enduku",
        "ela", "enti", "emiti", "ikkada", "akkada",
        "ivala", "repu", "kani", "mari", "kuda",
        "ante", "ledu", "avunu",
    },

    "Tamil": {
        "naan", "nanu", "nee", "neenga", "avan", "aval",
        "enga", "enakku", "unakku", "namma", "romba",
        "irukken", "irukku", "irundhan", "irundhen",
        "pesitu", "pesuren", "poga", "poiten",
        "poitten", "vandhen", "varuven", "varum",
        "enna", "ennaku", "yen", "epdi", "eppadi",
        "inga", "anga", "oda", "aama", "illa",
    },

    "Hindi": {
        "main", "mein", "mujhe", "mujhko", "tum",
        "aap", "hum", "ham", "hume", "mera", "meri",
        "mere", "tera", "teri", "aaj", "kal", "kya",
        "kyun", "kaise", "kaisa", "hai", "hain",
        "hoon", "tha", "thi", "gaya", "gayi",
        "gaye", "jaunga", "jaungi", "karna", "karta",
        "karti", "raha", "rahi", "rahe", "bahut",
        "accha", "achha", "nahi", "nahin", "haan",
        "wala", "wali", "wale",
    },

    "Bengali": {
        "ami", "tumi", "apni", "amra", "tomra",
        "amar", "tomar", "amader", "tomader",
        "aaj", "kal", "keno", "kemon", "ache",
        "achi", "achhe", "chilo", "gechi", "gechhi",
        "jabo", "jacchi", "korbo", "korchi",
        "korechi", "bhalo", "khub", "hya",
        "ekhane", "okhane",
    },
}


# ============================================================
# ENGLISH
# ============================================================

ENGLISH_WORDS = {
    "i", "you", "he", "she", "we", "they",
    "me", "my", "your", "his", "her", "our",
    "today", "tomorrow", "yesterday",
    "college", "school", "class", "friend",
    "friends", "home", "office", "work",
    "lunch", "dinner", "food", "morning",
    "evening", "night",
    "go", "went", "going", "come", "came",
    "coming", "eat", "ate", "eating",
    "want", "need", "like", "love",
    "good", "bad", "very", "really",
    "just", "now", "then", "here", "there",
    "what", "why", "how", "when", "where", "who",
    "yes", "no", "and", "or", "but", "so",
    "because", "with", "from", "to", "in", "on",
    "at", "for", "is", "am", "are", "was",
    "were", "be", "been", "will", "can",
    "could", "should", "have", "has", "had",
}


# ============================================================
# AMBIGUOUS WORDS
# ============================================================

AMBIGUOUS_WORDS = {
    "ki", "ke", "ka", "ku",
    "la", "na", "to", "ni",
    "lo", "um", "se", "me",
    "ma", "a", "i", "o",
}


# ============================================================
# EXPLICIT LANGUAGE MARKERS
# ============================================================

LANGUAGE_MARKERS = {
    "tamil": "Tamil",
    "telugu": "Telugu",
    "hindi": "Hindi",
    "bengali": "Bengali",
    "english": "English",
}


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

    if re.search(r"[\u0C00-\u0C7F]", token):
        return "Telugu"

    if re.search(r"[\u0B80-\u0BFF]", token):
        return "Tamil"

    if re.search(r"[\u0980-\u09FF]", token):
        return "Bengali"

    if re.search(r"[\u0900-\u097F]", token):
        return "Hindi"

    return None


# ============================================================
# SINGLE TOKEN DETECTION
# ============================================================

def detect_token_language(token: str) -> Dict:
    """
    Detect the language of a single token.

    Detection priority:
    1. Native Unicode script
    2. Explicit language marker
    3. Romanized Indic lexicon
    4. English lexicon
    5. Ambiguous-word handling
    6. Unknown
    """

    original = token
    token = token.lower().strip()

    # --------------------------------------------------------
    # Native script
    # --------------------------------------------------------

    native_language = detect_native_script(token)

    if native_language:
        return {
            "token": original,
            "language": native_language,
            "method": "native_script",
            "confidence": 1.0,
        }

    # --------------------------------------------------------
    # Explicit language marker
    #
    # Examples:
    # Tamil-la
    # Telugu-lo
    # Hindi-me
    # --------------------------------------------------------

    parts = token.split("-")

    for part in parts:
        if part in LANGUAGE_MARKERS:
            return {
                "token": original,
                "language": LANGUAGE_MARKERS[part],
                "method": "language_marker",
                "confidence": 1.0,
            }

    # --------------------------------------------------------
    # Romanized lexicon
    # --------------------------------------------------------

    candidates = set(parts)
    candidates.add(token)

    matches = {}

    for language, words in SEGMENT_LEXICON.items():

        matched_words = candidates.intersection(words)

        if not matched_words:
            continue

        score = 0

        for word in matched_words:

            # Longer words provide stronger evidence.
            if len(word) >= 5:
                score += 2

            elif len(word) >= 4:
                score += 1

        if score > 0:
            matches[language] = score

    if matches:

        best_language = max(
            matches,
            key=matches.get,
        )

        best_score = matches[best_language]

        return {
            "token": original,
            "language": best_language,
            "method": "romanized_lexicon",
            "confidence": min(1.0, best_score / 2.0),
        }

    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    if token in ENGLISH_WORDS:
        return {
            "token": original,
            "language": "English",
            "method": "english_lexicon",
            "confidence": 1.0,
        }

    # --------------------------------------------------------
    # Ambiguous token
    # --------------------------------------------------------

    if token in AMBIGUOUS_WORDS:
        return {
            "token": original,
            "language": "Unknown",
            "method": "ambiguous",
            "confidence": 0.0,
        }

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return {
        "token": original,
        "language": "Unknown",
        "method": "unknown",
        "confidence": 0.0,
    }


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_with_punctuation(text: str) -> List[str]:
    """
    Tokenize Romanized and native-script text.

    Keeps explicit language markers such as:
        Tamil-la
        Telugu-lo
    """

    return re.findall(
        r"[A-Za-z\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F]+(?:-[A-Za-z]+)?",
        text,
    )


# ============================================================
# CONTEXT RESOLUTION
# ============================================================

def resolve_ambiguous_tokens(
    segments: List[Dict],
) -> List[Dict]:
    """
    Resolve ambiguous tokens using nearby confident
    language predictions.

    Example:

        Nenu college ki vellanu

        Nenu     -> Telugu
        college  -> English
        ki       -> Unknown
        vellanu  -> Telugu

    The surrounding Telugu tokens provide contextual
    evidence that 'ki' is Telugu.
    """

    resolved = []

    indian_languages = {
        "Telugu",
        "Tamil",
        "Hindi",
        "Bengali",
    }

    for index, segment in enumerate(segments):

        # Already detected confidently.
        if segment["language"] != "Unknown":
            resolved.append(segment)
            continue

        token = segment["token"].lower()

        # Only resolve known ambiguous words.
        if token not in AMBIGUOUS_WORDS:
            resolved.append(segment)
            continue

        previous_language = None
        next_language = None

        # ----------------------------------------------------
        # Look backward for nearest known language.
        # ----------------------------------------------------

        for i in range(index - 1, -1, -1):

            language = segments[i]["language"]

            if language != "Unknown":
                previous_language = language
                break

        # ----------------------------------------------------
        # Look forward for nearest known language.
        # ----------------------------------------------------

        for i in range(index + 1, len(segments)):

            language = segments[i]["language"]

            if language != "Unknown":
                next_language = language
                break

        # ----------------------------------------------------
        # Both sides agree.
        # ----------------------------------------------------

        if (
            previous_language
            and next_language
            and previous_language == next_language
            and previous_language != "English"
        ):

            segment = segment.copy()

            segment["language"] = previous_language
            segment["method"] = "context_resolution"
            segment["confidence"] = 0.85

        # ----------------------------------------------------
        # Previous Indian-language token.
        # ----------------------------------------------------

        elif previous_language in indian_languages:

            segment = segment.copy()

            segment["language"] = previous_language
            segment["method"] = "context_resolution"
            segment["confidence"] = 0.70

        # ----------------------------------------------------
        # Next Indian-language token.
        # ----------------------------------------------------

        elif next_language in indian_languages:

            segment = segment.copy()

            segment["language"] = next_language
            segment["method"] = "context_resolution"
            segment["confidence"] = 0.70

        resolved.append(segment)

    return resolved


# ============================================================
# SEGMENT CODE-MIXED SENTENCE
# ============================================================

def segment_code_mix(text: str) -> List[Dict]:
    """
    Detect language for every token in a multilingual
    code-mixed sentence.
    """

    tokens = tokenize_with_punctuation(text)

    segments = []

    for token in tokens:

        result = detect_token_language(token)

        segments.append(result)

    # Resolve ambiguous words using context.
    segments = resolve_ambiguous_tokens(segments)

    return segments


# ============================================================
# PUBLIC LANGUAGE SEGMENT API
# ============================================================

def get_language_segments(text: str) -> List[Dict]:
    """
    Return token-level language segments.

    IMPORTANT:
    This function intentionally returns the full list of
    segment dictionaries instead of grouping tokens by
    language.

    This structure is required by:
        - normalization
        - semantic processing
        - response generation
        - voice pipeline

    Example output:

        [
            {
                "token": "Nenu",
                "language": "Telugu",
                "method": "romanized_lexicon",
                "confidence": 1.0
            },
            {
                "token": "today",
                "language": "English",
                "method": "english_lexicon",
                "confidence": 1.0
            }
        ]
    """

    return segment_code_mix(text)