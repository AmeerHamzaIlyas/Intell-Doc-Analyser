"""Extractor factory to route files to appropriate extractors."""
from pathlib import Path
from app.core.exceptions import UnsupportedFileTypeError
from app.services.extraction.base import BaseExtractor
from app.services.extraction.docx_extractor import DocxExtractor
from app.services.extraction.ocr_extractor import OCRExtractor
from app.services.extraction.pdf_extractor import PDFExtractor
from app.services.extraction.text_extractor import TextExtractor


def get_extractor(file_path: Path) -> BaseExtractor:
    """Return the corresponding extractor for the given file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PDFExtractor()
    elif ext == ".docx":
        return DocxExtractor()
    elif ext in (".txt", ".md"):
        return TextExtractor()
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        return OCRExtractor()
    else:
        raise UnsupportedFileTypeError(f"No extractor available for extension: '{ext}'")
