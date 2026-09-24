"""Document management endpoints: upload, list, get, delete, and stats."""
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from app.config import settings
from app.core.exceptions import DocumentIntelligenceException, DocumentNotFoundError
from app.core.security import validate_path_containment, verify_api_key
from app.logging_config import get_logger
from app.models.document import (
    DocumentChunk,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentMetadata,
)
from app.services.ai.embedding_service import embedding_service
from app.services.document_service import document_service
from app.services.storage.bm25_index import bm25_index
from app.services.storage.database import db
from app.services.storage.vector_store import vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])


def _index_document_chunks(chunks: List[DocumentChunk]):
    """Background or inline worker to index chunks into BM25 and VectorStore."""
    try:
        bm25_index.add_chunks(chunks)
        embedding_service.index_chunks(chunks)
        logger.info("Successfully indexed %d chunks in background", len(chunks))
    except Exception as e:
        logger.error("Failed indexing document chunks: %s", e)


@router.post("/upload", response_model=DocumentDetailResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload and ingest a PDF, DOCX, TXT, MD, or scanned image document."""
    content = await file.read()
    filename = file.filename or "uploaded_file"

    try:
        metadata, chunks = document_service.ingest_document(filename, content)
    except DocumentIntelligenceException:
        raise
    except Exception as e:
        logger.error("Upload error for '%s': %s", filename, e)
        raise HTTPException(status_code=500, detail=f"Internal document ingestion error: {e}")

    # Only index chunks if document is newly ingested (not a deduplicated upload)
    is_dup = getattr(metadata, "is_duplicate", False) or bool((metadata.extracted_entities or {}).get("is_duplicate", False))
    if not is_dup:
        if background_tasks:
            background_tasks.add_task(_index_document_chunks, chunks)
        else:
            _index_document_chunks(chunks)
    else:
        logger.info("Skipping vector/BM25 indexing for deduplicated document '%s'", metadata.document_id)

    return DocumentDetailResponse(
        metadata=metadata,
        chunk_count=len(chunks),
        section_count=len(set(c.section_title for c in chunks if c.section_title)),
        has_summary=bool(metadata.summary),
        has_vector_embeddings=True,
        is_duplicate=is_dup,
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List all ingested documents with operational stats."""
    docs = db.list_documents(limit=limit, offset=offset)
    total = db.count_documents()

    response_items = []
    for d in docs:
        chunks = db.get_chunks_for_document(d.document_id)
        sections = set(c.section_title for c in chunks if c.section_title)
        response_items.append(
            DocumentDetailResponse(
                metadata=d,
                chunk_count=len(chunks),
                section_count=len(sections),
                has_summary=bool(d.summary and "executive_summary" in d.summary),
                has_vector_embeddings=True,
            )
        )

    return DocumentListResponse(total=total, documents=response_items)


@router.get("/stats/overview")
def get_stats():
    """Retrieve system-wide analytics."""
    stats = db.get_system_stats()
    stats["vector_count"] = vector_store.count()
    stats["bm25_terms"] = len(bm25_index.doc_freqs)
    return stats


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document_detail(document_id: str):
    """Get single document metadata and structural metrics."""
    doc = db.get_document(document_id)
    if not doc:
        raise DocumentNotFoundError(f"Document '{document_id}' not found.")

    chunks = db.get_chunks_for_document(document_id)
    sections = set(c.section_title for c in chunks if c.section_title)

    return DocumentDetailResponse(
        metadata=doc,
        chunk_count=len(chunks),
        section_count=len(sections),
        has_summary=bool(doc.summary and "executive_summary" in doc.summary),
        has_vector_embeddings=True,
    )


@router.get("/{document_id}/chunks", response_model=List[DocumentChunk])
def get_document_chunks(document_id: str):
    """Retrieve all chunks for a document."""
    chunks = db.get_chunks_for_document(document_id)
    if not chunks:
        doc = db.get_document(document_id)
        if not doc:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
    return chunks


@router.delete("/{document_id}")
def delete_document(document_id: str):
    """Delete document, cascading its chunks, vector embeddings, and BM25 entries."""
    doc = db.get_document(document_id)
    if not doc:
        raise DocumentNotFoundError(f"Document '{document_id}' not found.")

    # 1. Remove from VectorStore
    vector_store.delete_document_vectors(document_id)

    # 2. Remove from BM25
    bm25_index.delete_document(document_id)

    # 3. Remove raw file from disk safely within UPLOAD_DIR
    clean_name = doc.filename
    uploaded_file = settings.UPLOAD_DIR / f"{document_id}_{clean_name}"
    try:
        validated_path = validate_path_containment(uploaded_file, settings.UPLOAD_DIR)
        validated_path.unlink(missing_ok=True)
    except Exception as path_err:
        logger.warning("Could not safely remove file %s: %s", uploaded_file, path_err)

    # 4. Remove from Database
    db.delete_document(document_id)

    return {
        "deleted": True,
        "document_id": document_id,
        "message": f"Document '{doc.filename}' and its indices were successfully deleted."
    }
