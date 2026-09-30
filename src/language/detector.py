import re
from typing import Dict, List

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


# IndicLID is used as a supporting signal.
# It is NOT treated as the final authority because
# sentence-level Romanized code-mixed text can be ambiguous.
roman_model = fasttext.load_model(MODEL_PATH)


# Romanized vocabulary for the languages currently
# targeted by MultiMix AI.
ROMANIZED_LEXICON = {

    "Telugu": {
        "nenu",
        "nuvvu",
        "meeru",
        "memu",
        "manam",
        "naaku",
        "neeku",
        "nee",
        "mana",
        "vellu",
        "vellanu",
        "vellava",
        "velthunna",
        "vastanu",
        "vastava",
        "vacchanu",
        "chesanu",
        "chestunna",
        "chestunnanu",
        "undi",
        "unnanu",
        "unnava",
        "enduku",
        "ela",
        "enti",
        "emiti",
        "ikkada",
        "akkada",
        "ivala",
        "repu",
        "kani",
        "mari",
        "kuda",
        "ante",
        "ledu",
        "avunu",
    },

    "Tamil": {
        "naan",
        "nanu",
        "neenga",
        "avan",
        "aval",
        "enga",
        "enakku",
        "unakku",
        "namma",
        "romba",
        "irukken",
        "irukku",
        "irundhan",
        "irundhen",
        "pesitu",
        "pesuren",
        "poga",
        "poiten",
        "poitten",
        "vandhen",
        "varuven",
        "varum",
        "enna",
        "ennaku",
        "yen",
        "epdi",
        "eppadi",
        "inga",
        "anga",
        "oda",
        "aama",
        "illa",
    },

    "Hindi": {
        "main",
        "mein",
        "mujhe",
        "mujhko",
        "tum",
        "aap",
        "hum",
        "ham",
        "hume",
        "mera",
        "meri",
        "mere",
        "tera",
        "teri",
        "aaj",
        "kal",
        "kya",
        "kyun",
        "kaise",
        "kaisa",
        "hai",
        "hain",
        "hoon",
        "tha",
        "thi",
        "gaya",
        "gayi",
        "gaye",
        "jaunga",
        "jaungi",
        "karna",
        "karta",
        "karti",
        "raha",
        "rahi",
        "rahe",
        "bahut",
        "accha",
        "achha",
        "nahi",
        "nahin",
        "haan",
        "wala",
        "wali",
        "wale",
        "par",
    },

    "Bengali": {
        "ami",
        "tumi",
        "apni",
        "amra",
        "tomra",
        "amar",
        "tomar",
        "amader",
        "tomader",
        "aaj",
        "keno",
        "kemon",
        "ache",
        "achi",
        "achhe",
        "chilo",
        "gechi",
        "gechhi",
        "jabo",
        "jacchi",
        "korbo",
        "korchi",
        "korechi",
        "bhalo",
        "khub",
        "hya",
        "ekhane",
        "okhane",
    },
}


# Common English words used for detecting English
# inside code-mixed sentences.
ENGLISH_WORDS = {
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "my",
    "your",
    "his",
    "her",
    "our",
    "today",
    "tomorrow",
    "yesterday",
    "college",
    "school",
    "class",
    "friend",
    "friends",
    "home",
    "office",
    "work",
    "lunch",
    "dinner",
    "food",
    "morning",
    "evening",
    "night",
    "go",
    "went",
    "going",
    "come",
    "came",
    "coming",
    "eat",
    "ate",
    "eating",
    "want",
    "need",
    "like",
    "love",
    "good",
    "bad",
    "very",
    "really",
    "just",
    "now",
    "then",
    "here",
    "there",
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "yes",
    "no",
    "and",
    "or",
    "but",
    "so",
    "because",
    "with",
    "from",
    "to",
    "in",
    "on",
    "at",
    "for",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "will",
    "can",
    "could",
    "should",
    "have",
    "has",
    "had",
}


# Very short words are highly ambiguous across
# Indian languages and English.
#
# They must NOT independently identify a language.
AMBIGUOUS_WORDS = {
    "ki",
    "ke",
    "ka",
    "ku",
    "la",
    "na",
    "to",
    "ni",
    "lo",
    "um",
    "se",
    "me",
    "ma",
    "a",
    "i",
    "o",
}


def detect_script_languages(text: str) -> List[str]:
    """
    Detect supported languages from native Unicode scripts.
    """

    detected = set()

    # Telugu
    if re.search(r"[\u0C00-\u0C7F]", text):
        detected.add("Telugu")

    # Tamil
    if re.search(r"[\u0B80-\u0BFF]", text):
        detected.add("Tamil")

    # Bengali
    if re.search(r"[\u0980-\u09FF]", text):
        detected.add("Bengali")

    # Hindi / Devanagari
    if re.search(r"[\u0900-\u097F]", text):
        detected.add("Hindi")

    return sorted(detected)


def predict_romanized(text: str, k: int = 5):
    """
    Run IndicLID and return its predictions.

    IndicLID is treated as supporting evidence,
    not as the final decision for code-mixed text.
    """

    text = text.strip()

    if not text:
        return []

    labels, probabilities = roman_model.predict(text, k=k)

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


def tokenize(text: str) -> List[str]:
    """
    Extract word-like tokens while preserving
    hyphenated forms such as Tamil-la.
    """

    return re.findall(
        r"[A-Za-z]+(?:-[A-Za-z]+)?",
        text.lower(),
    )


def lexical_scores(tokens: List[str]) -> Dict[str, float]:
    """
    Calculate Romanized-language evidence.

    Important:
    - Ambiguous short words are ignored.
    - Longer language-specific words receive stronger weight.
    - English is scored independently.
    """

    scores = {
        "Telugu": 0.0,
        "Tamil": 0.0,
        "Hindi": 0.0,
        "Bengali": 0.0,
        "English": 0.0,
    }

    for token in tokens:

        # Ignore highly ambiguous short words.
        if token in AMBIGUOUS_WORDS:
            continue

        # Support forms such as:
        # Tamil-la
        # Telugu-lo
        parts = token.split("-")

        candidates = set(parts)
        candidates.add(token)

        for language, words in ROMANIZED_LEXICON.items():

            matches = candidates.intersection(words)

            for match in matches:

                # Stronger evidence for longer words.
                if len(match) >= 5:
                    scores[language] += 2.0

                elif len(match) >= 4:
                    scores[language] += 1.0

        # English evidence.
        if token in ENGLISH_WORDS:
            scores["English"] += 1.0

    return scores


def analyze_code_mix(text: str) -> Dict:
    """
    Hybrid multilingual/code-mix analysis.

    Detection layers:

    1. Native Unicode script detection
    2. Romanized lexical evidence
    3. IndicLID supporting prediction
    4. Ambiguity filtering
    """

    # -----------------------------------------
    # STEP 1: Native-script detection
    # -----------------------------------------

    script_languages = detect_script_languages(text)

    # -----------------------------------------
    # STEP 2: Tokenization
    # -----------------------------------------

    tokens = tokenize(text)

    # -----------------------------------------
    # STEP 3: Romanized lexical evidence
    # -----------------------------------------

    lexical = lexical_scores(tokens)

    # -----------------------------------------
    # STEP 4: IndicLID prediction
    # -----------------------------------------

    roman_predictions = predict_romanized(
        text,
        k=10,
    )

    # -----------------------------------------
    # STEP 5: Combine evidence
    # -----------------------------------------

    languages = set(script_languages)

    # English:
    # One recognized English word is enough to
    # provide supporting evidence.
    if lexical["English"] >= 1:
        languages.add("English")

    # Indian Romanized languages:
    # Require stronger evidence to avoid false positives.
    for language in [
        "Telugu",
        "Tamil",
        "Hindi",
        "Bengali",
    ]:

        if lexical[language] >= 2:
            languages.add(language)

    # -----------------------------------------
    # STEP 6: IndicLID supporting evidence
    # -----------------------------------------

    #
    # IMPORTANT:
    #
    # We only accept IndicLID when:
    #
    # - confidence is high
    # - the language is one of our target languages
    # - there is no contradictory native-script evidence
    #
    # This prevents cases such as:
    #
    # Telugu Romanized text
    # -> IndicLID predicts Kannada
    #
    # from incorrectly adding Kannada.
    #

    for prediction in roman_predictions:

        language = prediction["language"]
        confidence = prediction["confidence"]

        if language not in {
            "Telugu",
            "Tamil",
            "Hindi",
            "Bengali",
            "English",
        }:
            continue

        if confidence < 0.85:
            continue

        # Don't allow IndicLID to introduce a different
        # Indian language when native script already
        # establishes another language.
        if script_languages:

            if language not in script_languages:

                # For native-script input, rely primarily
                # on the actual script and lexical evidence.
                continue

        languages.add(language)

    # -----------------------------------------
    # STEP 7: Final result
    # -----------------------------------------

    languages = set(languages)

    return {
        "input": text,
        "tokens": tokens,
        "languages": sorted(languages),
        "language_count": len(languages),
        "is_code_mixed": len(languages) > 1,
        "script_languages": script_languages,
        "lexical_scores": lexical,
        "roman_predictions": roman_predictions,
    }