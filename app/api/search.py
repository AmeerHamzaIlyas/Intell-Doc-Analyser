"""Search API endpoints for hybrid, dense vector, and BM25 sparse queries."""
from fastapi import APIRouter, HTTPException
from app.models.search_rag import SearchRequest, SearchResponse
from app.services.ai.embedding_service import embedding_service
from app.services.storage.hybrid_retriever import hybrid_retriever

router = APIRouter(prefix="/api/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
def execute_search(request: SearchRequest):
    """Execute hybrid semantic, dense vector, or sparse BM25 search."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    query_vector = None
    if request.mode in ("hybrid", "dense"):
        query_vector = embedding_service.embed_query(request.query)

    response = hybrid_retriever.search(
        query=request.query,
        query_vector=query_vector,
        top_k=request.top_k,
        mode=request.mode,
        document_ids=request.document_ids,
        min_score=request.min_score,
    )
    return response
