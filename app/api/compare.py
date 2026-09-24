"""Document comparison API endpoints."""
from fastapi import APIRouter, HTTPException
from app.models.search_rag import ComparisonRequest, ComparisonResponse
from app.services.ai.comparison_service import comparison_service

router = APIRouter(prefix="/api/compare", tags=["Comparison"])


@router.post("", response_model=ComparisonResponse)
def compare_documents(request: ComparisonRequest):
    """Compare two documents structurally and analyze their semantic differences."""
    if request.doc_id_a == request.doc_id_b:
        raise HTTPException(status_code=400, detail="Cannot compare a document with itself.")

    try:
        comparison = comparison_service.compare_documents(request.doc_id_a, request.doc_id_b)
        return comparison
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")
