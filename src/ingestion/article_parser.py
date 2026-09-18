"""Article parser module for the Azerbaijani Labour Code.

Parses extracted text into structured records: one per legal article (Maddə N),
preserving article number, section/chapter headings, and full text, while
performing Azerbaijani text normalization and suspicious extraction detection.
"""

import json
import re
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from src.config import settings
from src.ingestion.pdf_extractor import PageText, PDFExtractor
from src.ingestion.text_cleaner import (
    is_suspicious_text,
    normalize_azerbaijani_text,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ArticleRecord(BaseModel):
    """Structured representation of a parsed Azerbaijani legal article."""

    article_number: str = Field(
        ..., description="Article number, e.g. '70', '74', '70-1'"
    )
    title: str = Field(
        default="", description="Title or heading of the article"
    )
    section: str = Field(
        default="", description="Section (Bölmə) heading containing article"
    )
    chapter: str = Field(
        default="", description="Chapter (Fəsil) heading containing article"
    )
    content: str = Field(
        ..., description="Normalized legal body text of the article"
    )
    full_text: str = Field(
        ..., description="Full formatted representation with heading & content"
    )
    page_start: int = Field(
        default=1, description="First page number where article appears"
    )
    page_end: int = Field(
        default=1, description="Last page number where article appears"
    )
    is_suspicious: bool = Field(
        default=False, description="Flag indicating potential extraction flaw"
    )
    suspicious_reason: str = Field(
        default="", description="Explanation if marked suspicious"
    )


class ArticleParser:
    """Parses raw extracted text or pages into structured ArticleRecords."""

    ARTICLE_REGEX = re.compile(
        r"^(?:MADDƏ|Maddə|maddə)\s+(\d+(?:[.-]\d+)?)\s*[.:\-–]?\s*(.*)$"
    )
    HEADING_FOOTNOTE_REGEX = re.compile(
        r"(?<=[A-Za-zÇƏĞİÖŞÜçəğıöşü])\d{1,3}$"
    )
    SECTION_REGEX = re.compile(
        r"^(?:(?:[IVXLCDM\d]+\s+)?BÖLMƏ|"
        r"BÖLMƏ\s+[IVXLCDM\d]+|Bölmə\s+[IVXLCDM\d]+)"
        r"[.:\-–]?\s*(.*)$",
        re.IGNORECASE
    )
    CHAPTER_REGEX = re.compile(
        r"^(?:(?:[IVXLCDM\d]+\s+)?FƏSİL|"
        r"FƏSİL\s+[IVXLCDM\d]+|Fəsil\s+[IVXLCDM\d]+)"
        r"[.:\-–]?\s*(.*)$",
        re.IGNORECASE
    )

    def __init__(self, min_article_length: int = 40):
        """Initializes parser with validation threshold.

        Args:
            min_article_length: Threshold for flagging short articles.
        """
        self.min_article_length = min_article_length

    def parse_pages(self, pages: List[PageText]) -> List[ArticleRecord]:
        """Parses PageText list into structured ArticleRecord instances.

        Args:
            pages: List of PageText objects from PDFExtractor.

        Returns:
            List[ArticleRecord]: List of parsed and normalized articles.
        """
        articles: List[ArticleRecord] = []
        current_section = ""
        current_chapter = ""

        current_article_num: Optional[str] = None
        current_article_title_lines: List[str] = []
        current_article_body_lines: List[str] = []
        current_page_start = 1
        current_page_end = 1
        is_reading_title = False
        footnote_regex = self.HEADING_FOOTNOTE_REGEX

        parsing_completed = False
        for page in pages:
            if parsing_completed:
                break
            lines = page.text.split("\n")
            for raw_line in lines:
                line = normalize_azerbaijani_text(raw_line).strip()
                if not line:
                    continue

                # Stop when reaching amendment history endnotes
                if "İSTİFADƏ OLUNMUŞ MƏNBƏ" in line.upper():
                    parsing_completed = True
                    break

                # Check for Section heading
                sec_match = self.SECTION_REGEX.match(line)
                if sec_match:
                    current_section = line
                    continue

                # Check for Chapter heading
                chap_match = self.CHAPTER_REGEX.match(line)
                if chap_match:
                    current_chapter = line
                    continue

                # Check for Article heading
                art_match = self.ARTICLE_REGEX.match(line)
                if art_match:
                    # Flush previous article if one was in progress
                    if current_article_num is not None:
                        record = self._build_record(
                            article_num=current_article_num,
                            title_lines=current_article_title_lines,
                            body_lines=current_article_body_lines,
                            section=current_section,
                            chapter=current_chapter,
                            page_start=current_page_start,
                            page_end=current_page_end,
                        )
                        articles.append(record)

                    # Start new article
                    current_article_num = art_match.group(1).strip()
                    current_article_title_lines = []
                    current_article_body_lines = []
                    current_page_start = page.page_number
                    current_page_end = page.page_number

                    trailing_title = art_match.group(2).strip()
                    if trailing_title:
                        cleaned_title = re.sub(
                            r"\s*\d{1,4}$", "", trailing_title
                        ).strip()
                        current_article_title_lines.append(cleaned_title)
                        # Check if title requires continuation
                        needs_continuation = bool(re.search(
                            r"(?:nın|nin|nun|nün|ın|in|un|ün|"
                            r"və|və ya|ilə|üzrə|haqqında)$",
                            cleaned_title.lower()
                        ))
                        is_reading_title = needs_continuation
                    else:
                        is_reading_title = True
                    continue

                # We are inside an active article
                if current_article_num is not None:
                    current_page_end = page.page_number

                    # Check if line continues title or starts body
                    if is_reading_title:
                        is_clause = bool(re.match(
                            r"^(?:\d+[.)]|[a-zçəğıöşü][.)])", line
                        ))
                        cleaned_line = re.sub(
                            r"\s*\d{1,4}$", "", line
                        ).strip()
                        is_verb = bool(re.search(
                            r"(?:edilir|verilir|olunur|bilər|"
                            r"təşkil edir|sayılır)[.:]?$",
                            cleaned_line.lower()
                        ))
                        if (
                            not is_clause
                            and not is_verb
                            and len(cleaned_line.split()) <= 6
                        ):
                            current_article_title_lines.append(cleaned_line)
                            is_reading_title = False
                        else:
                            is_reading_title = False
                            current_article_body_lines.append(line)
                    else:
                        current_article_body_lines.append(line)

        # Flush final article
        if current_article_num is not None:
            record = self._build_record(
                article_num=current_article_num,
                title_lines=current_article_title_lines,
                body_lines=current_article_body_lines,
                section=current_section,
                chapter=current_chapter,
                page_start=current_page_start,
                page_end=current_page_end,
            )
            articles.append(record)

        logger.info(f"Parsed {len(articles)} articles successfully.")
        return articles

    def _build_record(
        self,
        article_num: str,
        title_lines: List[str],
        body_lines: List[str],
        section: str,
        chapter: str,
        page_start: int,
        page_end: int,
    ) -> ArticleRecord:
        """Constructs and normalizes an ArticleRecord instance."""
        raw_title = " ".join(title_lines).strip()
        raw_body = "\n".join(body_lines).strip()

        norm_title = normalize_azerbaijani_text(raw_title)
        norm_body = normalize_azerbaijani_text(raw_body)
        norm_sec = normalize_azerbaijani_text(section)
        norm_chap = normalize_azerbaijani_text(chapter)

        # Full formatted text representation
        if norm_title:
            full_text = f"Maddə {article_num}. {norm_title}\n\n{norm_body}"
        else:
            full_text = f"Maddə {article_num}.\n\n{norm_body}"

        suspicious, reason = is_suspicious_text(
            norm_body, min_length=self.min_article_length
        )

        return ArticleRecord(
            article_number=article_num,
            title=norm_title,
            section=norm_sec,
            chapter=norm_chap,
            content=norm_body,
            full_text=full_text,
            page_start=page_start,
            page_end=page_end,
            is_suspicious=suspicious,
            suspicious_reason=reason,
        )

    def save_records(
        self,
        articles: List[ArticleRecord],
        output_json: Path,
        output_jsonl: Path,
    ) -> None:
        """Saves parsed articles to JSON and JSONL formats with UTF-8 encoding.

        Args:
            articles: List of ArticleRecord instances.
            output_json: Path to target JSON file.
            output_jsonl: Path to target JSONL file.
        """
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        records_dict = [a.model_dump() for a in articles]
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(records_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(articles)} articles to {output_json}")

        # Save JSONL
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for record in records_dict:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(articles)} articles to {output_jsonl}")


def run_ingestion(pdf_path: Optional[Path] = None) -> List[ArticleRecord]:
    """CLI / programmatic entrypoint for PDF ingestion and parsing.

    Args:
        pdf_path: Optional path to PDF. If not provided, searches data/raw/.

    Returns:
        List[ArticleRecord]: Parsed articles.

    Raises:
        FileNotFoundError: If no PDF is found.
    """
    if pdf_path is None:
        pdf_files = list(settings.DATA_RAW_DIR.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {settings.DATA_RAW_DIR}. "
                "Please place the Azerbaijani Labour Code PDF into data/raw/."
            )
        pdf_path = pdf_files[0]
        logger.info(f"Auto-selected PDF: {pdf_path}")

    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract_pages()

    parser = ArticleParser()
    articles = parser.parse_pages(pages)

    parser.save_records(
        articles,
        output_json=settings.ARTICLES_JSON,
        output_jsonl=settings.ARTICLES_JSONL,
    )

    return articles


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    custom_pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_ingestion(custom_pdf)
