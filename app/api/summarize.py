"""Summarization API endpoints for multi-tier document summaries."""
from fastapi import APIRouter, HTTPException, Query
from app.models.document import DocumentSummaryModel
from app.services.ai.summary_service import summary_service

router = APIRouter(prefix="/api/summarize", tags=["Summarization"])


@router.post("/{document_id}", response_model=DocumentSummaryModel)
def generate_summary(
    document_id: str,
    force_refresh: bool = Query(default=False, description="Re-run summarization bypassing cache"),
):
    """Generate or retrieve a multi-tier summary (Executive, Key Findings, Sections, Actions)."""
    try:
        summary = summary_service.summarize_document(document_id, force_refresh=force_refresh)
        return summary
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
