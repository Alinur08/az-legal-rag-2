"""Lightweight tests for centralized legal disclaimer handling."""

import unittest
from unittest.mock import patch

from src.config import settings
from src.guardrails.disclaimer import add_legal_disclaimer


class TestLegalDisclaimer(unittest.TestCase):
    def test_default_language_is_azerbaijani(self):
        self.assertEqual(
            add_legal_disclaimer("Cavab"),
            "Cavab\n\n" + settings.LEGAL_DISCLAIMER_AZ,
        )

    def test_explicit_azerbaijani(self):
        self.assertEqual(
            add_legal_disclaimer("Cavab", "az"),
            "Cavab\n\n" + settings.LEGAL_DISCLAIMER_AZ,
        )

    def test_english(self):
        self.assertEqual(
            add_legal_disclaimer("Answer", "en"),
            "Answer\n\n" + settings.LEGAL_DISCLAIMER_EN,
        )

    def test_exactly_two_newline_separator(self):
        answer = "Cavab"
        result = add_legal_disclaimer(answer)
        self.assertEqual(
            result[len(answer):-len(settings.LEGAL_DISCLAIMER_AZ)], "\n\n"
        )

    def test_azerbaijani_disclaimer_not_duplicated(self):
        text = "Cavab\n\n" + settings.LEGAL_DISCLAIMER_AZ
        self.assertEqual(add_legal_disclaimer(text, "az"), text)

    def test_english_disclaimer_not_duplicated(self):
        text = "Answer\n\n" + settings.LEGAL_DISCLAIMER_EN
        self.assertEqual(add_legal_disclaimer(text, "en"), text)

    def test_unsupported_language(self):
        with self.assertRaisesRegex(ValueError, "Unsupported language.*Use 'az' or 'en'"):
            add_legal_disclaimer("Cavab", "fr")

    def test_original_text_preserved(self):
        answer = "  Əmək hüquqları haqqında cavab.\nİkinci sətir.  \n"
        self.assertEqual(
            add_legal_disclaimer(answer),
            answer + "\n\n" + settings.LEGAL_DISCLAIMER_AZ,
        )

    def test_uses_current_config_values(self):
        for language, attribute in (
            ("az", "LEGAL_DISCLAIMER_AZ"),
            ("en", "LEGAL_DISCLAIMER_EN"),
        ):
            with self.subTest(language=language):
                with patch.object(settings, attribute, "Test config value"):
                    self.assertEqual(
                        add_legal_disclaimer("Answer", language),
                        "Answer\n\nTest config value",
                    )
                    self.assertEqual(
                        add_legal_disclaimer("Test config value", language),
                        "Test config value",
                    )


if __name__ == "__main__":
    unittest.main()
