"""PDF text extraction module using PyMuPDF.

Justification for PyMuPDF (fitz) over pdfplumber:
1. Performance: PyMuPDF is backed by high-performance C MuPDF libraries,
   extracting multi-hundred-page legal texts in seconds rather than minutes.
2. Encoding & Diacritics: Accurately parses UTF-8 font tables, preserving
   Azerbaijani Latin characters (ə, ğ, ı, ö, ş, ü, ç) without dropouts.
3. Layout Preservation: Supports block-level extraction to retain reading
   order and line break boundaries critical for legal parsing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pymupdf

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageText:
    """Represents extracted text and metadata for a single PDF page.

    Attributes:
        page_number: 1-indexed page number.
        text: Raw extracted text of the page.
    """
    page_number: int
    text: str


class PDFExtractor:
    """Extracts raw text and metadata from PDF files using PyMuPDF."""

    def __init__(self, pdf_path: Path):
        """Initializes the PDFExtractor with the target file path.

        Args:
            pdf_path: Path to the target PDF file.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

    def extract_pages(self) -> List[PageText]:
        """Extracts text page by page from the PDF.

        Returns:
            List[PageText]: List of PageText objects containing page numbers
            and raw extracted content.
        """
        pages: List[PageText] = []
        logger.info(f"Opening PDF document: {self.pdf_path.name}")

        try:
            with pymupdf.open(self.pdf_path) as doc:
                total_pages = len(doc)
                logger.info(
                    f"Total pages in {self.pdf_path.name}: {total_pages}"
                )

                for page_idx in range(total_pages):
                    page = doc[page_idx]
                    page_text = page.get_text("text")
                    pages.append(
                        PageText(page_number=page_idx + 1, text=page_text)
                    )
        except Exception as err:
            logger.error(f"Error reading PDF {self.pdf_path}: {err}")
            raise

        logger.info(f"Successfully extracted {len(pages)} pages.")
        return pages

    def extract_full_text(self) -> str:
        """Extracts and concatenates text across all pages with page markers.

        Returns:
            str: Combined text with delimited page markers.
        """
        pages = self.extract_pages()
        text_parts = []
        for p in pages:
            text_parts.append(f"\n--- [PAGE {p.page_number}] ---\n{p.text}")
        return "".join(text_parts)
