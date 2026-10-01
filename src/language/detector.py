from pathlib import Path
import re
from typing import Dict, List, Set, Tuple

import fasttext

from src.config import INDICLID_MODEL_PATH


MODEL_PATH = str(INDICLID_MODEL_PATH)

LANGUAGE_NAMES = {
    "eng": "English",
    "tel": "Telugu",
    "tam": "Tamil",
    "hin": "Hindi",
    "ben": "Bengali",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "asm": "Assamese",
    "ori": "Odia",
    "kok": "Konkani",
    "mni": "Manipuri",
}


# ============================================================
# SAFE INDICLID LAZY LOADER
# ============================================================

_roman_model = None
_indiclid_attempted = False


def get_roman_model():
    """
    Lazy-load the IndicLID fastText model safely if the model file exists on disk.
    Returns the loaded model instance, or None if the model is unavailable.
    """
    global _roman_model, _indiclid_attempted
    if _roman_model is not None:
        return _roman_model
    if _indiclid_attempted:
        return None

    _indiclid_attempted = True
    model_file = Path(MODEL_PATH)
    if model_file.exists() and model_file.stat().st_size > 1000:
        try:
            _roman_model = fasttext.load_model(str(model_file))
            return _roman_model
        except Exception:
            _roman_model = None
            return None
    return None


class _RomanModelProxy:
    """Backwards-compatible proxy for roman_model that delegates to get_roman_model()."""

    def __getattr__(self, name):
        m = get_roman_model()
        if m is None:
            raise AttributeError(f"IndicLID fastText model is unavailable at {MODEL_PATH}")
        return getattr(m, name)

    def predict(self, *args, **kwargs):
        m = get_roman_model()
        if m is None:
            return ((), ())
        return m.predict(*args, **kwargs)

    def __bool__(self):
        return get_roman_model() is not None


roman_model = _RomanModelProxy()


# ============================================================
# NATIVE SCRIPT DETECTION
# ============================================================

SCRIPT_RANGES = {
    "Telugu": re.compile(r"[\u0C00-\u0C7F]"),
    "Tamil": re.compile(r"[\u0B80-\u0BFF]"),
    "Bengali": re.compile(r"[\u0980-\u09FF]"),
    "Hindi": re.compile(r"[\u0900-\u097F]"),
}


def detect_script_languages(text: str) -> List[str]:
    """Detect supported languages from native Unicode scripts."""
    detected = set()
    for language, pattern in SCRIPT_RANGES.items():
        if pattern.search(text):
            detected.add(language)
    return sorted(detected)


# ============================================================
# GENERALIZED ROMANIZED LEXICON
# ============================================================

ROMANIZED_LEXICON = {
    "Telugu": {
        "nenu", "nuvvu", "meeru", "memu", "manam", "naa", "naaku", "neeku", "nee", "nannu", "natho",
        "maaku", "meeku", "mee", "mana", "manaku", "athadu", "athanu", "aame", "vaallu", "vaaru", "idi", "adi",
        "peru", "illu", "pani", "annam", "annamu", "neellu", "roju", "samayam", "ooru", "pelli", "daggara",
        "vellu", "vellanu", "vellava", "vellali", "velthunna", "velthunnadu", "velthunnaru", "vellipoyanu",
        "poyanu", "poyadu", "poya", "vastanu", "vastava", "vacchanu", "vachanu", "raavali",
        "chesanu", "chestunna", "chestunnanu", "chestunnav", "chestunnadu", "chestunnaru", "cheyali", "cheyi",
        "undi", "unnanu", "unnava", "unnadu", "unnaru", "untanu", "unta", "tinnanu", "tinu",
        "choodu", "choosanu", "cheppu", "cheppanu", "ledu", "avunu", "kadu",
        "enduku", "ela", "enti", "emiti", "em", "emi", "ikkada", "akkada", "ivala", "repu", "ninna",
        "kani", "mari", "kuda", "kooda", "ante", "inka", "malli", "chala", "koncham", "eppudu", "ekkadiki", "evaru",
    },

    "Tamil": {
        "naan", "nanu", "en", "ennoda", "ennudaiya", "enakku", "nee", "un", "unnoda", "unakku",
        "neenga", "neengal", "avan", "aval", "avanga", "avargal", "namma", "engal", "enga", "idhu", "adhu",
        "per", "peyar", "peru", "veedu", "velai", "saapadu", "thanni", "naal", "neram", "oor",
        "pesitu", "pesuren", "pesu", "pesina", "pesa", "poga", "ponen", "poneenga", "poiten", "poitten", "poitu", "poren",
        "vandhen", "varuven", "vaanga", "panra", "panren", "panreenga", "pannu", "panniten", "pannita",
        "irundhan", "irundhen", "irundhadhu", "irundhanga", "irukken", "irukku", "irukkeenga",
        "solren", "solli", "solla", "paathen", "paaru", "saaptiya", "saapten", "mudiyum", "mudiyadhu",
        "theriyum", "theriyadhu", "illa", "illai", "aama",
        "enna", "ennada", "yen", "epdi", "eppadi", "eppo", "enga", "enge", "inga", "anga",
        "romba", "konjam", "inniku", "nalaiki", "oda",
    },

    "Hindi": {
        "main", "mein", "mera", "meri", "mere", "mujhe", "mujhko", "tum", "tumhara", "tumhari", "tumhare", "tumhe",
        "aap", "aapka", "aapki", "aapke", "hum", "ham", "hamara", "hamari", "hamare", "hume",
        "yeh", "woh", "uska", "uski", "uske", "unka", "unki", "unke", "log",
        "naam", "ghar", "kaam", "khana", "paani", "din", "waqt", "samay", "bhai", "dost", "yaar",
        "hai", "hain", "hoon", "hun", "tha", "thi", "the",
        "karna", "karta", "karti", "karte", "kar", "karo", "kiya",
        "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "jaunga", "jaungi", "jaenge", "jao",
        "aao", "aaya", "aayi", "aaye", "aana", "bolo", "bola", "boli", "dekh", "dekha", "dekho",
        "hoga", "hogi", "honge", "hona",
        "aaj", "kal", "parso", "kya", "kyun", "kyon", "kaise", "kaisa", "kaisi", "kab", "kahan", "kidhar", "kaun",
        "bahut", "accha", "achha", "acchi", "nahi", "nahin", "haan", "wala", "wali", "wale", "bhi", "aur", "par", "lekin",
    },

    "Bengali": {
        "ami", "amar", "amake", "amra", "amader", "tumi", "tomar", "tomake", "tomra", "tomader",
        "apni", "apnar", "apnake", "se", "tar", "take", "tader", "eta", "ota", "ora", "era",
        "naam", "bari", "kaaj", "khabar", "bhaat", "jol", "din", "somoy", "bondhu",
        "ache", "achi", "achhe", "chilo", "gechi", "gechhi", "jabo", "jacchi", "korbo", "korchi", "korcho", "korche", "korechi", "koro",
        "bolo", "bolchi", "khabo", "kheyechi", "dekho", "dekhchi", "ashbo", "ashchi",
        "aaj", "kaal", "keno", "kemon", "kothay", "kobe", "ke", "khub", "bhalo", "anek", "ekhane", "okhane", "hya",
    },
}

ENGLISH_WORDS = {
    "i", "you", "he", "she", "we", "they", "me", "my", "your", "his", "her", "our",
    "today", "tomorrow", "yesterday", "college", "school", "class", "friend", "friends",
    "home", "office", "work", "lunch", "dinner", "food", "morning", "evening", "night",
    "go", "went", "going", "come", "came", "coming", "eat", "ate", "eating",
    "want", "need", "like", "love", "good", "bad", "very", "really", "just", "now", "then",
    "here", "there", "what", "why", "how", "when", "where", "who", "yes", "no", "and", "or",
    "but", "so", "because", "with", "from", "to", "in", "on", "at", "for", "is", "am", "are",
    "was", "were", "be", "been", "will", "can", "could", "should", "have", "has", "had",
    "name", "doing", "speaking", "talking", "said", "say", "see",
}

# Standalone short ambiguous words: never grant standalone language identification
AMBIGUOUS_WORDS = {
    "ki", "ke", "ka", "ku", "la", "na", "to", "ni", "lo", "um", "se", "me", "ma", "a", "i", "o", "di", "da",
}

LANGUAGE_MARKERS = {
    "tamil": "Tamil",
    "telugu": "Telugu",
    "hindi": "Hindi",
    "bengali": "Bengali",
    "english": "English",
}

# Multi-token context collocations: disambiguates short pronouns and nouns
CONTEXT_BIGRAMS = {
    "Telugu": {
        ("na", "peru"), ("naa", "peru"), ("nee", "peru"), ("mee", "peru"),
        ("ivala", "em"), ("em", "chestunnav"), ("emi", "chestunnav"),
        ("ki", "vellanu"), ("ki", "poyanu"), ("ki", "vacchanu"),
    },
    "Tamil": {
        ("en", "peru"), ("en", "peyar"), ("ennoda", "peru"), ("un", "peru"),
        ("enna", "panra"), ("college", "ponen"), ("college", "poiten"),
    },
    "Hindi": {
        ("mera", "naam"), ("meri", "naam"), ("mere", "naam"), ("tera", "naam"), ("aapka", "naam"),
        ("hum", "log"), ("aap", "log"), ("tum", "log"),
        ("kya", "kar"), ("kar", "rahe"), ("ja", "raha"), ("raha", "hoon"), ("rahi", "hoon"), ("rahe", "ho"),
        ("ki", "gaye"),
    },
    "Bengali": {
        ("amar", "naam"), ("tomar", "naam"), ("apnar", "naam"),
        ("ki", "korcho"), ("ki", "korchi"), ("college", "jabo"), ("college", "gechi"),
    },
    "English": {
        ("my", "name"), ("your", "name"), ("his", "name"), ("her", "name"), ("name", "is"),
        ("what", "are"), ("are", "you"), ("you", "doing"), ("went", "to"), ("to", "college"),
    },
}

# Language-specific morphological endings for unseen Romanized words
MORPHOLOGY_PATTERNS = {
    "Telugu": [
        re.compile(r"^[a-z]+(?:tunna|tunnanu|tunnav|tunnadu|tunnaru|tundi)$"),
        re.compile(r"^[a-z]{2,}(?:llanu|yanu|sanu|chanu|nanu)$"),
        re.compile(r"^[a-z]{3,}(?:yali|lali|vali)$"),
        re.compile(r"^[a-z]{3,}(?:stanu|stundi|thanu)$"),
    ],
    "Tamil": [
        re.compile(r"^[a-z]{2,}(?:suren|nren|nra|rren|lren|reenga)$"),
        re.compile(r"^[a-z]{2,}(?:ndhan|ndhen|ponen|poiten|poitten|ndhadhu|ndhanga)$"),
        re.compile(r"^[a-z]{2,}(?:situ|nitu|oitu|pitu)$"),
        re.compile(r"^[a-z]{2,}(?:akku|ukku)$"),
    ],
    "Hindi": [
        re.compile(r"^[a-z]{2,}(?:unga|ungi|enge|ega|egi)$"),
        re.compile(r"^[a-z]{2,}(?:wala|wali|wale)$"),
        re.compile(r"^[a-z]{2,}(?:karta|karti|karte|bolta|bolti|bolte|jaata|jaati|jaate)$"),
    ],
    "Bengali": [
        re.compile(r"^[a-z]{2,}(?:rbo|jabo|sbo|rbe|rben)$"),
        re.compile(r"^[a-z]{2,}(?:rcho|rchi|rche|rchen|cchi)$"),
        re.compile(r"^[a-z]{2,}(?:rechi|reche|rchilo|chilo)$"),
        re.compile(r"^[a-z]{2,}(?:mader|maderke|uder|ader)$"),
    ],
}


# ============================================================
# TOKENIZATION & EVIDENCE SCORING
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Extract word-like tokens while preserving hyphenated markers (e.g. Tamil-la).
    """
    return re.findall(
        r"[A-Za-z\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F]+(?:-[A-Za-z]+)?",
        text,
    )


def lexical_scores(tokens: List[str]) -> Dict[str, float]:
    """
    Calculate generalized Romanized-language and multi-signal evidence.

    Signals utilized:
    1. Lexicon matches (with length-weighted scoring)
    2. Collocation / Bigram context matches
    3. Language morphology / suffix patterns
    4. Explicit hyphenated markers (e.g. Tamil-la)
    5. English vocabulary matches
    6. Proper name neutralization (names never generate language evidence)
    7. Ambiguous short words suppression
    """
    scores = {
        "Telugu": 0.0,
        "Tamil": 0.0,
        "Hindi": 0.0,
        "Bengali": 0.0,
        "English": 0.0,
    }

    if not tokens:
        return scores

    tokens_lower = [t.lower() for t in tokens]

    # Neutralize proper name candidates directly following name markers
    name_indices: Set[int] = set()
    for idx, tok in enumerate(tokens_lower):
        if tok in {"peru", "peyar", "naam", "name"} and idx + 1 < len(tokens_lower):
            name_indices.add(idx + 1)

    # 1. Collocation / Bigram scoring
    for i in range(len(tokens_lower) - 1):
        bg = (tokens_lower[i], tokens_lower[i + 1])
        for lang, bigrams in CONTEXT_BIGRAMS.items():
            if bg in bigrams:
                scores[lang] += 2.5

    # 2. Token-level analysis
    for idx, tok in enumerate(tokens_lower):
        if idx in name_indices:
            # Proper names must never be classified as language evidence
            continue

        # Check explicit hyphenated markers (e.g. Tamil-la, Telugu-lo)
        parts = tok.split("-")
        for part in parts:
            if part in LANGUAGE_MARKERS:
                scores[LANGUAGE_MARKERS[part]] += 3.0

        # Standalone ambiguous words do not provide independent evidence
        if tok in AMBIGUOUS_WORDS:
            continue

        # Lexical matching
        for lang, words in ROMANIZED_LEXICON.items():
            if tok in words:
                scores[lang] += 2.0 if len(tok) >= 5 else 1.5

        # English matching
        if tok in ENGLISH_WORDS:
            scores["English"] += 1.0

        # Morphological pattern matching
        for lang, patterns in MORPHOLOGY_PATTERNS.items():
            for pat in patterns:
                if pat.match(tok):
                    scores[lang] += 1.5
                    break

    return scores


def predict_romanized(text: str, k: int = 5):
    """
    Run IndicLID and return its predictions.
    IndicLID is treated as supporting evidence, not as the sole decider.
    If the model is unavailable, returns an empty list.
    """
    text = text.strip()
    if not text:
        return []

    model = get_roman_model()
    if model is None:
        return []

    try:
        labels, probabilities = model.predict(text, k=k)
    except Exception:
        return []

    results = []
    for label, probability in zip(labels, probabilities):
        code = label.replace("__label__", "").split("_")[0]
        results.append(
            {
                "language": LANGUAGE_NAMES.get(code, code),
                "confidence": float(probability),
            }
        )

    return results


# ============================================================
# HYBRID MULTILINGUAL & CODE-MIX ANALYSIS
# ============================================================

def analyze_code_mix(text: str) -> Dict:
    """
    Hybrid multilingual / code-mix detection.

    Combines:
    1. Native Unicode script detection
    2. Generalized Romanized lexical evidence
    3. Morphological suffix patterns
    4. Multi-token context / collocations
    5. Ambiguous-word handling
    6. IndicLID supporting prediction (when available)
    """
    # 1. Native-script detection
    script_languages = detect_script_languages(text)

    # 2. Tokenization
    tokens = tokenize(text)

    # 3. Multi-signal evidence scoring
    lexical = lexical_scores(tokens)

    # 4. IndicLID prediction (if model available)
    roman_predictions = predict_romanized(text, k=10)

    # 5. Combine evidence
    languages = set(script_languages)

    # English: One recognized English word is sufficient supporting evidence
    if lexical.get("English", 0.0) >= 1.0:
        languages.add("English")

    # Indian Romanized languages: Require score >= 2.0
    for language in ["Telugu", "Tamil", "Hindi", "Bengali"]:
        if lexical.get(language, 0.0) >= 2.0:
            languages.add(language)

    # 6. IndicLID supporting evidence (when available)
    tokens_lower = [t.lower() for t in tokens]
    has_non_ambiguous = any(t not in AMBIGUOUS_WORDS for t in tokens_lower)

    if has_non_ambiguous:
        for prediction in roman_predictions:
            language = prediction["language"]
            confidence = prediction["confidence"]

            if language not in {"Telugu", "Tamil", "Hindi", "Bengali", "English"}:
                continue

            if confidence < 0.85:
                continue

            # Don't allow IndicLID to introduce an Indian language conflicting with native script
            if script_languages and language not in script_languages:
                continue

            # English: only accept IndicLID English prediction if supported by actual English vocabulary
            # (prevents IndicLID eng_Latn bias on Romanized proper names / out-of-vocabulary words)
            if language == "English" and lexical.get("English", 0.0) < 1.0:
                continue

            languages.add(language)

    # 7. Final structured response
    final_languages = sorted(languages)

    return {
        "input": text,
        "tokens": tokens,
        "languages": final_languages,
        "language_count": len(final_languages),
        "is_code_mixed": len(final_languages) > 1,
        "script_languages": script_languages,
        "lexical_scores": lexical,
        "roman_predictions": roman_predictions,
        "indiclid_available": get_roman_model() is not None,
    }