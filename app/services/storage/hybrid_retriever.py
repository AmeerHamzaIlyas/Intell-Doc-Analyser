"""Hybrid retrieval engine combining Dense Vector search and Sparse BM25 via Reciprocal Rank Fusion."""
import time
from typing import Dict, List, Optional
from app.config import settings
from app.logging_config import get_logger
from app.models.search_rag import SearchResponse, SearchResultChunk
from app.services.storage.bm25_index import BM25Index, bm25_index
from app.services.storage.database import Database, db
from app.services.storage.vector_store import VectorStore, vector_store

logger = get_logger(__name__)


class HybridRetriever:
    """Combines dense vector embeddings and sparse BM25 scores using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        db_instance: Optional[Database] = None,
        vec_store: Optional[VectorStore] = None,
        bm25_idx: Optional[BM25Index] = None,
        rrf_k: int = 60,
    ):
        self.db = db_instance or db
        self.vector_store = vec_store or vector_store
        self.bm25 = bm25_idx or bm25_index
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
        mode: str = "hybrid",
        document_ids: Optional[List[str]] = None,
        min_score: float = 0.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> SearchResponse:
        """Perform hybrid, dense, or sparse search across indexed document chunks."""
        start_time = time.perf_counter()

        dense_results: List[tuple] = []
        sparse_results: List[tuple] = []

        # 1. Dense retrieval (if vector provided and mode is hybrid or dense)
        if mode in ("hybrid", "dense") and query_vector is not None:
            # Over-fetch for re-ranking
            dense_k = top_k * 3 if mode == "hybrid" else top_k
            dense_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=dense_k,
                document_ids=document_ids,
                min_score=min_score,
            )

        # 2. Sparse retrieval (if mode is hybrid or sparse)
        if mode in ("hybrid", "sparse"):
            sparse_k = top_k * 3 if mode == "hybrid" else top_k
            sparse_results = self.bm25.search(
                query=query,
                top_k=sparse_k,
                document_ids=document_ids,
                min_score=min_score,
            )

        # 3. Combine scores based on selected mode
        combined_scores: Dict[str, Dict[str, float]] = {}

        if mode == "dense":
            for cid, score in dense_results:
                combined_scores[cid] = {"score": score, "method": "dense"}
        elif mode == "sparse":
            for cid, score in sparse_results:
                combined_scores[cid] = {"score": score, "method": "sparse"}
        else:  # Hybrid RRF
            # Add dense ranks
            for rank, (cid, score) in enumerate(dense_results):
                rrf_score = dense_weight / (self.rrf_k + rank + 1)
                combined_scores[cid] = {
                    "score": rrf_score,
                    "dense_score": score,
                    "sparse_score": 0.0,
                    "method": "hybrid_dense",
                }

            # Add sparse ranks
            for rank, (cid, score) in enumerate(sparse_results):
                rrf_score = sparse_weight / (self.rrf_k + rank + 1)
                if cid in combined_scores:
                    combined_scores[cid]["score"] += rrf_score
                    combined_scores[cid]["sparse_score"] = score
                    combined_scores[cid]["method"] = "hybrid_rrf"
                else:
                    combined_scores[cid] = {
                        "score": rrf_score,
                        "dense_score": 0.0,
                        "sparse_score": score,
                        "method": "hybrid_sparse",
                    }

        # Sort combined results descending by score
        ranked_items = sorted(
            combined_scores.items(), key=lambda item: item[1]["score"], reverse=True
        )[:top_k]

        if not ranked_items:
            latency = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                query=query,
                mode=mode,
                total_results=0,
                results=[],
                latency_ms=round(latency, 2),
            )

        # 4. Fetch chunk text and document metadata
        chunk_ids = [cid for cid, _ in ranked_items]
        chunks = self.db.get_chunks_by_ids(chunk_ids)
        chunk_map = {c.chunk_id: c for c in chunks}

        # Cache document filenames
        doc_filename_cache: Dict[str, str] = {}
        for c in chunks:
            if c.document_id not in doc_filename_cache:
                doc = self.db.get_document(c.document_id)
                doc_filename_cache[c.document_id] = doc.filename if doc else "unknown"

        result_chunks: List[SearchResultChunk] = []
        for cid, meta in ranked_items:
            if cid not in chunk_map:
                continue
            c = chunk_map[cid]
            result_chunks.append(
                SearchResultChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    filename=doc_filename_cache.get(c.document_id, "unknown"),
                    page_number=c.page_number,
                    section_title=c.section_title or "General",
                    text=c.text,
                    score=round(meta["score"], 4),
                    retrieval_method=meta["method"],
                    char_start=c.char_start,
                    char_end=c.char_end,
                )
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        return SearchResponse(
            query=query,
            mode=mode,
            total_results=len(result_chunks),
            results=result_chunks,
            latency_ms=round(latency, 2),
        )


# Global hybrid retriever instance
hybrid_retriever = HybridRetriever()
