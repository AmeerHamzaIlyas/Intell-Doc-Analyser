"""Text cleaning and normalization service."""
import re
import unicodedata
from typing import List
from app.services.extraction.base import ExtractedPage


class TextCleaner:
    """Normalizes Unicode, repairs broken words, removes running headers/footers."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Perform standard text cleaning pipeline on a text string.
        
        - Strips zero-width characters, BOM, and non-printable control characters
        - Normalizes non-breaking and unusual Unicode spaces
        - Standardizes line breaks and removes excessive empty lines
        - Reconnects hyphenated word breaks across lines
        - Performs Unicode NFKC composition
        """
        if not text:
            return ""

        # 1. Normalize line endings
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Strip zero-width non-printable characters and byte-order-marks (BOM)
        cleaned = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", cleaned)

        # 3. Strip ASCII control characters (preserving tab \t and newline \n)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

        # 4. Normalize non-breaking and special Unicode spaces to standard space
        cleaned = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", cleaned)

        # 5. Unicode normalization (NFKC: Compatibility Decomposition followed by Canonical Composition)
        cleaned = unicodedata.normalize("NFKC", cleaned)

        # 6. Repair hyphenated line breaks (e.g., 'inter-\nnational' -> 'international')
        cleaned = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", cleaned)

        # 7. Normalize horizontal whitespace and tabs
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)

        # 8. Remove excessive blank lines (more than 2 newlines -> 2 newlines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    @classmethod
    def clean_pages(cls, pages: List[ExtractedPage]) -> List[ExtractedPage]:
        """Clean pages and remove running headers/footers repeated across pages."""
        if not pages:
            return []

        # Find candidate headers and footers across pages
        # Look at the first 2 lines and last 2 lines of each page
        first_lines = []
        last_lines = []
        for p in pages:
            lines = [line.strip() for line in p.text.splitlines() if line.strip()]
            if lines:
                first_lines.append(lines[0])
            if len(lines) > 1:
                last_lines.append(lines[-1])

        # If a line appears in more than 60% of pages (and >= 3 pages), it's likely a header/footer
        repeated_lines = set()
        threshold = max(2, int(len(pages) * 0.6))
        
        for pool in (first_lines, last_lines):
            counts = {}
            for line in pool:
                if len(line) < 100:  # Reasonable header/footer length
                    counts[line] = counts.get(line, 0) + 1
            for line, cnt in counts.items():
                if cnt >= threshold:
                    repeated_lines.add(line)

        cleaned_pages: List[ExtractedPage] = []
        for p in pages:
            lines = [line.strip() for line in p.text.splitlines() if line.strip()]
            if len(lines) <= 2:
                filtered_lines = lines
            else:
                filtered_lines = [l for l in lines if l not in repeated_lines]
            cleaned_page_text = cls.clean_text("\n".join(filtered_lines))
            cleaned_pages.append(
                ExtractedPage(
                    page_number=p.page_number,
                    text=cleaned_page_text,
                    tables=p.tables,
                    section_hints=p.section_hints,
                    has_ocr=p.has_ocr
                )
            )

        return cleaned_pages
