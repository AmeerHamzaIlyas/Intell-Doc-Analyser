"""Document and Chunk schemas and data models."""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from app.core.constants import STATUS_QUEUED


class DocumentMetadata(BaseModel):
    """Metadata describing an uploaded and processed document."""
    document_id: str
    filename: str
    original_filename: str
    file_type: str
    mime_type: str
    file_size: int
    file_hash: str
    page_count: int = 1
    word_count: int = 0
    char_count: int = 0
    reading_time_minutes: float = 0.0
    detected_language: str = "en"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = STATUS_QUEUED
    error_message: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[Dict[str, Any]] = None
    is_duplicate: bool = False


class DocumentChunk(BaseModel):
    """A semantic chunk of a document preserving provenance."""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int = 1
    section_title: Optional[str] = "General"
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    """A structural section detected within a document."""
    title: str
    level: int = 1
    page_number: int = 1
    char_start: int = 0
    char_end: int = 0


class DocumentSummaryModel(BaseModel):
    """Multi-tier document summary."""
    document_id: str
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    section_summaries: Dict[str, str] = Field(default_factory=dict)
    action_items: List[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DocumentDetailResponse(BaseModel):
    """Detailed response for a single document."""
    metadata: DocumentMetadata
    chunk_count: int = 0
    section_count: int = 0
    has_summary: bool = False
    has_vector_embeddings: bool = False
    is_duplicate: bool = False


class DocumentListResponse(BaseModel):
    """Paginated or listed documents."""
    total: int
    documents: List[DocumentDetailResponse]
