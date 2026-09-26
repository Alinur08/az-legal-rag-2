"""Append legal disclaimers from the centralized configuration."""

from src.config import settings


def add_legal_disclaimer(text: str, language: str = "az") -> str:
    """Append the selected disclaimer unless its exact text is already present.

    Preserve the original text and separate it from the disclaimer with two
    newlines. Only 'az' and 'en' are supported.
    """
    if language == "az":
        disclaimer = settings.LEGAL_DISCLAIMER_AZ
    elif language == "en":
        disclaimer = settings.LEGAL_DISCLAIMER_EN
    else:
        raise ValueError(f"Unsupported language: {language!r}. Use 'az' or 'en'.")

    if disclaimer in text:
        return text
    return text + "\n\n" + disclaimer
