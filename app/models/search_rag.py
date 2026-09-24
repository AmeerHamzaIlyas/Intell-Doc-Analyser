"""Search, RAG, and Document Comparison schemas."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from app.models.document import DocumentMetadata


class SearchRequest(BaseModel):
    """Payload for semantic and hybrid search."""
    query: str = Field(..., min_length=1, max_length=4000, description="Search query text")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document filter")
    top_k: int = Field(default=5, ge=1, le=50, description="Max results to return")
    mode: str = Field(default="hybrid", description="Search mode: hybrid, dense, or sparse")
    min_score: float = Field(default=0.0, ge=0.0, description="Minimum score threshold")


class SearchResultChunk(BaseModel):
    """A matched chunk returned from search."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_title: Optional[str] = None
    text: str
    score: float
    retrieval_method: str  # "dense", "sparse", "hybrid_rrf"
    char_start: int = 0
    char_end: int = 0


class SearchResponse(BaseModel):
    """Search results response with performance metrics."""
    query: str
    mode: str
    total_results: int
    results: List[SearchResultChunk]
    latency_ms: float


class Citation(BaseModel):
    """Source provenance citation for a RAG response."""
    citation_id: int
    document_id: str
    filename: str
    page_number: int
    section_title: Optional[str] = "General"
    snippet: str
    confidence_score: float = 1.0


class RAGRequest(BaseModel):
    """Payload for question answering with RAG."""
    query: str = Field(..., min_length=1, max_length=4000, description="User question")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional target documents")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        max_length=50,
        description="Previous messages (capped at 50 turns)"
    )


class RAGResponse(BaseModel):
    """Source-grounded RAG response with exact citations."""
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    grounding_status: str = "GROUNDED"  # "GROUNDED", "PARTIAL", "NO_EVIDENCE"
    referenced_chunks: List[SearchResultChunk] = Field(default_factory=list)
    latency_ms: float = 0.0


class ComparisonRequest(BaseModel):
    """Payload to compare two documents."""
    doc_id_a: str = Field(..., description="First document ID")
    doc_id_b: str = Field(..., description="Second document ID")


class ComparisonResponse(BaseModel):
    """Side-by-side structural and semantic comparison response."""
    doc_a: DocumentMetadata
    doc_b: DocumentMetadata
    structural_diff: Dict[str, Any]
    semantic_analysis: Dict[str, Any]
    generated_at: str
