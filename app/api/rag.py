"""RAG Q&A endpoints with citation resolution."""
from fastapi import APIRouter, HTTPException
from app.models.search_rag import RAGRequest, RAGResponse
from app.services.ai.rag_service import rag_service

router = APIRouter(prefix="/api/rag", tags=["RAG"])


@router.post("/query", response_model=RAGResponse)
def answer_rag_query(request: RAGRequest):
    """Answer a user question grounded strictly in document chunks with citations."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = rag_service.answer_query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {e}")
