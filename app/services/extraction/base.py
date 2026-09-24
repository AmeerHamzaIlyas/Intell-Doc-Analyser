"""Base extractor interfaces and data structures."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Extracted text and structural content of a single document page."""
    page_number: int
    text: str
    tables: List[List[List[str]]] = Field(default_factory=list)
    section_hints: List[str] = Field(default_factory=list)
    has_ocr: bool = False


class ExtractedDocument(BaseModel):
    """Full extraction result for a document."""
    pages: List[ExtractedPage]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    total_pages: int
    full_text: str
    title: Optional[str] = None
    author: Optional[str] = None


class BaseExtractor(ABC):
    """Abstract interface for format-specific extractors."""

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text, pages, and metadata from a file."""
        pass
