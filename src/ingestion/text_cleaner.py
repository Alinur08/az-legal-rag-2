"""Azerbaijani text cleaning and normalization module.

Handles Azerbaijani diacritic normalization (ə, ğ, ı, ö, ş, ü, ç), fixes
hyphenated word splits across line breaks, removes non-printable artifacts,
and detects suspicious or corrupted text chunks.
"""

import re
import unicodedata
from typing import Tuple


# Mapping of Cyrillic lookalikes / legacy glyphs to Azerbaijani Latin Unicode
AZERBAIJANI_CHAR_MAP = {
    "\u04D9": "ə",  # Cyrillic small schwa -> Latin small schwa
    "\u04D8": "Ə",  # Cyrillic capital schwa -> Latin capital schwa
    "\u018F": "Ə",  # Latin capital schwa (canonical)
    "\u0259": "ə",  # Latin small schwa (canonical)
    "\u00AD": "",   # Soft hyphen -> remove
    "\u200B": "",   # Zero-width space -> remove
    "\uFEFF": "",   # Byte order mark -> remove
    "\u2019": "'",  # Right single quotation mark -> standard apostrophe
    "\u2018": "'",  # Left single quotation mark -> standard apostrophe
    "\u201C": '"',  # Left double quotation mark -> standard quote
    "\u201D": '"',  # Right double quotation mark -> standard quote
    "\u00A0": " ",  # Non-breaking space -> regular space
}


def normalize_azerbaijani_text(text: str) -> str:
    """Normalizes Azerbaijani text and resolves encoding/spacing artifacts.

    Args:
        text: Raw extracted text string.

    Returns:
        Cleaned, normalized text string with proper diacritics.
    """
    if not text:
        return ""

    # Replace legacy/Cyrillic lookalike characters
    for old_char, new_char in AZERBAIJANI_CHAR_MAP.items():
        text = text.replace(old_char, new_char)

    # Unicode canonical decomposition and recomposition (NFC)
    text = unicodedata.normalize("NFC", text)

    # Rejoin words broken across line breaks with hyphens:
    # e.g., "müqavilə-\n sinin" or "işəgötürən-\n dən" -> "müqaviləsinin"
    text = re.sub(r"([a-zçəğıöşüA-ZÇƏĞİÖŞÜ])-\s*\n\s*([a-zçəğıöşüA-ZÇƏĞİÖŞÜ])",
                  r"\1\2", text)

    # Rejoin soft line breaks within sentences while keeping paragraph breaks
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(re.sub(r"[ \t]+", " ", line).strip())

    # Collapse multiple blank lines into at most two newlines
    joined_text = "\n".join(cleaned_lines)
    joined_text = re.sub(r"\n{3,}", "\n\n", joined_text)

    # Normalize spacing around punctuation: e.g. "Maddə 70 . " -> "Maddə 70. "
    joined_text = re.sub(r"\s+([.,;:!?])", r"\1", joined_text)

    return joined_text.strip()


def is_suspicious_text(text: str,
                       min_length: int = 40) -> Tuple[bool, str]:
    """Inspects text for signs of extraction corruption or incompleteness.

    Args:
        text: Extracted text chunk.
        min_length: Minimum expected character length for a legal article.

    Returns:
        Tuple of (is_suspicious: bool, reason: str).
    """
    if not text or len(text.strip()) == 0:
        return True, "Empty text block"

    cleaned = text.strip()
    if len(cleaned) < min_length:
        return True, f"Abnormally short length ({len(cleaned)} chars)"

    # Check for unicode replacement character
    if "\ufffd" in text:
        return True, "Contains unicode replacement character (U+FFFD)"

    # Check for excessive question marks indicating unmapped characters
    question_mark_ratio = text.count("?") / len(cleaned)
    if question_mark_ratio > 0.15:
        return True, f"High question mark density ({question_mark_ratio:.1%})"

    # Check for alphabetic character density
    alpha_chars = sum(1 for c in cleaned if c.isalpha())
    alpha_ratio = alpha_chars / len(cleaned)
    if alpha_ratio < 0.35:
        return True, f"Low alphabetic character ratio ({alpha_ratio:.1%})"

    return False, ""
