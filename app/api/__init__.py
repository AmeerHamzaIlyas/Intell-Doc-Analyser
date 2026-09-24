"""API routers package."""
from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.api.rag import router as rag_router
from app.api.summarize import router as summarize_router
from app.api.compare import router as compare_router

__all__ = [
    "health_router",
    "documents_router",
    "search_router",
    "rag_router",
    "summarize_router",
    "compare_router",
]
