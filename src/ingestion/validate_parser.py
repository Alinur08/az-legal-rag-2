"""Validation module for spot-checking parsed Azerbaijani Labour Code articles.

Verifies that key articles (e.g. Maddə 70, Maddə 74) are accurately extracted,
contain expected legal domain terms, preserve Azerbaijani diacritics, and are
not flagged as suspicious.
"""

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _az_lower(s: str) -> str:
    """Lowercases a string with Azerbaijani-aware İ→i and I→ı mapping."""
    return unicodedata.normalize(
        "NFC", s.replace("İ", "i").replace("I", "ı").lower()
    )

# Known anchor articles to spot check in the Azerbaijani Labour Code
SPOT_CHECK_EXPECTATIONS: Dict[str, Dict[str, List[str]]] = {
    "70": {
        "title_keywords": ["əsasları", "işəgötürən"],
        "body_keywords": ["müqavilə", "ixtisar", "vəzifə"],
    },
    "74": {
        "title_keywords": ["iradəsindən", "asılı", "olmayan"],
        "body_keywords": ["hərbi", "məhkəmə", "qüvvəyə"],
    },
}


def validate_articles_file(
    json_path: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """Validates parsed articles JSON against expected legal anchor articles.

    Args:
        json_path: Path to articles.json. Defaults to settings.ARTICLES_JSON.

    Returns:
        Tuple of (all_passed: bool, report_lines: List[str]).
    """
    path = json_path or settings.ARTICLES_JSON
    if not path.is_file():
        return False, [f"Error: Processed articles file not found at {path}"]

    with open(path, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except Exception as err:
            return False, [f"Error decoding JSON at {path}: {err}"]

    article_map = {str(r.get("article_number")): r for r in records}
    report: List[str] = []
    all_passed = True

    report.append(f"Loaded {len(records)} articles from {path.name}")
    report.append("-" * 60)

    # Check for suspicious items overall
    suspicious_count = sum(1 for r in records if r.get("is_suspicious"))
    if suspicious_count > 0:
        report.append(
            f"WARNING: {suspicious_count} article(s) flagged as suspicious."
        )
    else:
        report.append("OK: No suspicious articles detected.")

    # Spot check specific articles
    for art_num, expectations in SPOT_CHECK_EXPECTATIONS.items():
        if art_num not in article_map:
            report.append(
                f"FAIL: Maddə {art_num} missing from parsed records."
            )
            all_passed = False
            continue

        item = article_map[art_num]
        title = _az_lower(item.get("title", ""))
        content = _az_lower(item.get("content", ""))
        is_suspicious = item.get("is_suspicious", False)

        if is_suspicious:
            report.append(
                f"FAIL: Maddə {art_num} is flagged suspicious: "
                f"{item.get('suspicious_reason')}"
            )
            all_passed = False
            continue

        # Check title keywords
        missing_title_kw = [
            kw for kw in expectations["title_keywords"] if kw not in title
        ]
        # Check body keywords
        missing_body_kw = [
            kw for kw in expectations["body_keywords"] if kw not in content
        ]

        if missing_title_kw:
            report.append(
                f"FAIL: Maddə {art_num} title missing expected keywords: "
                f"{missing_title_kw}"
            )
            all_passed = False
        elif missing_body_kw:
            report.append(
                f"FAIL: Maddə {art_num} body missing expected keywords: "
                f"{missing_body_kw}"
            )
            all_passed = False
        else:
            report.append(
                f"PASS: Maddə {art_num} - '{item.get('title')[:50]}...' "
                f"({len(item.get('content', ''))} chars, pages "
                f"{item.get('page_start')}-{item.get('page_end')})"
            )

    report.append("-" * 60)
    report.append(f"Validation Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed, report


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    custom_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    success, lines = validate_articles_file(custom_path)
    for line in lines:
        print(line)
    sys.exit(0 if success else 1)
