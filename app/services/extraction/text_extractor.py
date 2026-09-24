"""Plaintext and Markdown document extractor."""
from pathlib import Path
from app.core.exceptions import ExtractionError
from app.services.extraction.base import BaseExtractor, ExtractedDocument, ExtractedPage


class TextExtractor(BaseExtractor):
    """Extracts content from TXT and Markdown files with encoding resilience."""

    def extract(self, file_path: Path) -> ExtractedDocument:
        if file_path.stat().st_size == 0:
            return ExtractedDocument(
                pages=[ExtractedPage(page_number=1, text="", tables=[], section_hints=[])],
                metadata={"source": str(file_path), "encoding": "empty"},
                total_pages=1,
                full_text="",
                title=file_path.stem
            )

        encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1", "cp1252"]
        content: str = ""
        used_enc: str = "utf-8"
        
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                used_enc = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue
                
        if not content and file_path.stat().st_size > 0:
            raise ExtractionError(f"Failed to decode text file with standard encodings: {file_path}")

        # Derive title from first line or filename
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        title = lines[0].lstrip("#").strip() if lines else file_path.stem

        page = ExtractedPage(
            page_number=1,
            text=content,
            tables=[],
            section_hints=[line.lstrip("#").strip() for line in lines if line.startswith("#")]
        )

        return ExtractedDocument(
            pages=[page],
            metadata={"source": str(file_path), "encoding": used_enc},
            total_pages=1,
            full_text=content,
            title=title
        )
