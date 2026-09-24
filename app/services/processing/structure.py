"""Document structure and heading detection."""
import re
from typing import List, Tuple
from app.models.document import DocumentSection
from app.services.extraction.base import ExtractedPage

# Patterns for heading detection
HEADING_PATTERNS = [
    # Markdown headings: # Heading 1, ## Heading 2
    (re.compile(r"^(#{1,6})\s+(.+)$"), "markdown"),
    # Numbered headings: 1. Introduction, 2.1 Background, A. Appendix
    (re.compile(r"^((?:[0-9]{1,2}\.){1,3}|[A-Z]\.)\s+([A-Z][A-Za-z0-9\s,:-]{2,80})$"), "numbered"),
    # Uppercase headings: EXECUTIVE SUMMARY, FINANCIAL HIGHLIGHTS
    (re.compile(r"^([A-Z0-9\s,:-]{4,60})$"), "uppercase"),
]


class StructureDetector:
    """Detects headings, table boundaries, and hierarchical sections."""

    @classmethod
    def detect_sections(cls, pages: List[ExtractedPage]) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        global_offset = 0

        for page in pages:
            lines = page.text.splitlines()
            for line in lines:
                trimmed = line.strip()
                line_len = len(line) + 1  # include newline
                if not trimmed:
                    global_offset += line_len
                    continue

                level, title = cls._classify_heading(trimmed)
                if title:
                    sections.append(
                        DocumentSection(
                            title=title,
                            level=level,
                            page_number=page.page_number,
                            char_start=global_offset,
                            char_end=global_offset + len(trimmed)
                        )
                    )
                global_offset += line_len

        return sections

    @staticmethod
    def _classify_heading(line: str) -> Tuple[int, str]:
        """Determine if a line is a heading and its hierarchy level (1-6)."""
        # 1. Markdown syntax
        md_match = HEADING_PATTERNS[0][0].match(line)
        if md_match:
            hashes, title = md_match.groups()
            return len(hashes), title.strip()

        # 2. Numbered heading
        num_match = HEADING_PATTERNS[1][0].match(line)
        if num_match:
            prefix, title = num_match.groups()
            dots = prefix.count(".")
            level = min(dots + 1, 4)
            return level, f"{prefix} {title}".strip()

        # 3. Uppercase heading
        if len(line) < 60 and line.isupper() and any(c.isalpha() for c in line):
            # Avoid single short noise words like 'PAGE 1' or 'TABLE'
            if not line.startswith("PAGE ") and len(line) > 3:
                return 2, line.title()

        return 0, ""
