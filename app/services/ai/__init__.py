"""AI services package: Gemini client, embeddings, RAG, summarization, and comparison."""
from app.services.ai.gemini_client import GeminiClient, gemini_client
from app.services.ai.embedding_service import EmbeddingService, embedding_service
from app.services.ai.rag_service import RAGService, rag_service
from app.services.ai.summary_service import SummaryService, summary_service
from app.services.ai.comparison_service import ComparisonService, comparison_service

__all__ = [
    "GeminiClient", "gemini_client",
    "EmbeddingService", "embedding_service",
    "RAGService", "rag_service",
    "SummaryService", "summary_service",
    "ComparisonService", "comparison_service",
]
