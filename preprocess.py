"""
Tier 2: Preprocessing Pipeline (preprocess.py)
- Applies Unicode NFKC normalization
- Strips zero-width and control characters
- Maps Cyrillic/Greek look-alike homoglyphs to canonical Latin characters
- Counts homoglyph substitutions as a risk feature (homoglyph_hits)
- Identifies language with langdetect (graceful fallback if unavailable)
"""

import re
import unicodedata

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception

# Zero-width and invisible formatting characters commonly used to evade filters
ZERO_WIDTH_RE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]"
)

# Control characters except standard whitespace (\t, \n, \r)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Common Cyrillic and Greek look-alike homoglyphs mapped to Latin ASCII
HOMOGLYPH_MAP = {
    "\u0430": "a",  # Cyrillic small a
    "\u0410": "A",  # Cyrillic capital A
    "\u0435": "e",  # Cyrillic small ie
    "\u0415": "E",  # Cyrillic capital IE
    "\u043e": "o",  # Cyrillic small o
    "\u041e": "O",  # Cyrillic capital O
    "\u0440": "p",  # Cyrillic small er
    "\u0420": "P",  # Cyrillic capital ER
    "\u0441": "c",  # Cyrillic small es
    "\u0421": "C",  # Cyrillic capital ES
    "\u0443": "y",  # Cyrillic small u
    "\u0423": "Y",  # Cyrillic capital U
    "\u0445": "x",  # Cyrillic small ha
    "\u0425": "X",  # Cyrillic capital HA
    "\u0456": "i",  # Cyrillic small byelorussian-ukrainian i
    "\u0406": "I",  # Cyrillic capital byelorussian-ukrainian I
    "\u0455": "s",  # Cyrillic small dze
    "\u0405": "S",  # Cyrillic capital DZE
    "\u0458": "j",  # Cyrillic small je
    "\u03b1": "a",  # Greek small alpha
    "\u03bf": "o",  # Greek small omicron
    "\u03c1": "p",  # Greek small rho
    "\u03b5": "e",  # Greek small epsilon
    "\u03ba": "k",  # Greek small kappa
    "\u03bd": "v",  # Greek small nu
}


def normalize(text: str):
    """
    Normalize input prompt against character-level obfuscation.

    Returns:
        tuple[str, int, str]: (normalized_text, homoglyph_hits, detected_language)
    """
    if not text:
        return "", 0, "en"

    # 1. Strip zero-width and control characters (count them toward obfuscation hits)
    zw_matches = len(ZERO_WIDTH_RE.findall(text))
    cleaned = ZERO_WIDTH_RE.sub("", text)
    cleaned = CONTROL_CHARS_RE.sub("", cleaned)

    # 2. Count and replace Cyrillic/Greek homoglyphs before NFKC folding
    homoglyph_hits = zw_matches
    chars = []
    for ch in cleaned:
        if ch in HOMOGLYPH_MAP:
            chars.append(HOMOGLYPH_MAP[ch])
            homoglyph_hits += 1
        else:
            # Also check full-width ASCII characters (U+FF01 to U+FF5E)
            if "\uff01" <= ch <= "\uff5e":
                homoglyph_hits += 1
            chars.append(ch)
    cleaned = "".join(chars)

    # 3. Apply Unicode NFKC normalization (folds full-width & compatibility chars)
    normalized = unicodedata.normalize("NFKC", cleaned)

    # Collapse excessive whitespace while preserving single newlines
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()

    # 4. Language detection
    lang = "en"
    if detect is not None and len(normalized) >= 5:
        try:
            lang = detect(normalized)
        except LangDetectException:
            lang = "en"

    return normalized, homoglyph_hits, lang
